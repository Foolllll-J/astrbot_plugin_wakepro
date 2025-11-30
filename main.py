import asyncio
import json
import time
import random
from pydantic import BaseModel, ConfigDict, Field
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import At
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter
from astrbot.core.star.star_handler import star_handlers_registry
from astrbot.api import logger
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.utils.session_waiter import session_waiter, SessionController
from .sentiment import Sentiment
from .similarity import Similarity


# ==================== 常量定义 ====================

# AstrBot 内置指令列表
BUILT_CMDS = [
    "llm", "t2i", "tts", "sid", "op", "wl",
    "dashboard_update", "alter_cmd", "provider", "model",
    "plugin", "plugin ls", "new", "switch", "rename",
    "del", "reset", "history", "persona", "tool ls",
    "key", "websearch", "help",
]

# 合并延迟期间最多合并的消息数量（防止轰炸攻击）
MAX_MERGE_MESSAGES = 10


# ==================== 数据模型 ====================

class MemberState(BaseModel):
    """群成员状态"""
    uid: str                                              # 用户ID
    silence_until: float = 0.0                            # 沉默截止时间（时间戳）
    last_request: float = 0.0                             # 最后一次发送LLM请求的时间（时间戳）
    last_response: float = 0.0                            # 最后一次LLM响应的时间（时间戳）
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)  # 异步锁
    in_merging: bool = False                              # 是否正在消息合并状态中
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GroupState(BaseModel):
    """群组状态"""
    gid: str                                              # 群组ID
    members: dict[str, MemberState] = Field(default_factory=dict)  # 成员状态字典
    shutup_until: float = 0.0                             # 群组闭嘴截止时间（时间戳）


class StateManager:
    """状态管理器 - 管理所有群组和成员的状态"""
    
    _groups: dict[str, GroupState] = {}
    
    @classmethod
    def get_group(cls, gid: str) -> GroupState:
        """获取或创建群组状态"""
        if gid not in cls._groups:
            cls._groups[gid] = GroupState(gid=gid)
        return cls._groups[gid]


@register(
    "astrbot_plugin_wakepro",
    "Zhalslar&Foolllll",
    "更强大的唤醒增强插件",
    "v1.1.7",
)
class WakeProPlugin(Star):
    """
    WakePro 插件主类
    
    设计流程:
    1. 消息级别(on_group_msg): 只处理黑白名单、内置指令屏蔽、唤醒判断
    2. LLM请求级别(on_llm_request): 处理所有检测和防护
       ├── 沉默触发检测(辱骂、闭嘴、AI检测)
       ├── 防护机制(沉默状态、闭嘴状态、请求CD)
       └── 消息合并
    """
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.sent = Sentiment()

    # ==================== 消息级别: 仅基础检查 ====================
    
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1)
    async def on_group_msg(self, event: AstrMessageEvent):
        """
        【第一层: 消息级别 - 基础检查和唤醒判断】

        处理流程:
        1. 全局屏蔽检查(黑白名单、权限)
        2. 内置指令屏蔽
        3. 检查是否已被其他插件处理
        4. 唤醒条件判断(决定是否调用LLM)
        """
        # 提取基本信息
        chain = event.get_messages()
        bid: str = event.get_self_id()
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        msg: str = event.message_str
        g: GroupState = StateManager.get_group(gid)

        # 只处理文本消息
        if not msg:
            return
        cmd = msg.split(" ", 1)[0]

        # ========== 1. 全局屏蔽检查(快速失败) ==========
        
        # 过滤 bot 自己的消息
        if uid == bid:
            return
        
        # 群聊白名单检查
        if self.conf["group_whitelist"] and gid not in self.conf["group_whitelist"]:
            return
        
        # 群聊黑名单检查
        if gid in self.conf["group_blacklist"] and not event.is_admin():
            event.stop_event()
            return
        
        # 用户黑名单检查
        if uid in self.conf.get("user_blacklist", []):
            event.stop_event()
            return

        # ========== 2. 内置指令屏蔽 ==========
        
        if self.conf["block_builtin"]:
            if not event.is_admin() and event.message_str in BUILT_CMDS:
                logger.debug(f"用户({uid})触发内置指令，已屏蔽")
                event.stop_event()
                return
        
        # ========== 3. 唤醒条件判断(决定是否标记为LLM请求) ==========
        
        # 初始化或获取用户状态
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        member = g.members[uid]
        now = time.time()

        # --- 唤醒条件判断 ---
        
        wake = event.is_at_or_wake_command  # 是否唤醒
        reason = "at_or_cmd"                 # 唤醒原因

        # 1. 提及唤醒：消息中包含特定关键词
        if not wake and self.conf["mention_wake"]:
            names = [n for n in self.conf["mention_wake"] if n]
            for n in names:
                if n and n in msg:
                    wake = True
                    reason = f"提及唤醒({n})"
                    break

        # 2. 唤醒延长：在上次LLM响应后的延长窗口内
        if (
            not wake
            and self.conf["wake_extend"]
            and (now - member.last_response) <= int(self.conf["wake_extend"] or 0)
        ):
            wake = True
            reason = "唤醒延长"

        # 3. 话题相关性唤醒：与最近对话内容相关
        if not wake and self.conf["relevant_wake"]:
            if bmsgs := await self._get_history_msg(event, count=5):
                for bmsg in bmsgs:
                    simi = Similarity.cosine(msg, bmsg, gid)
                    if simi > self.conf["relevant_wake"]:
                        wake = True
                        reason = f"话题相关性{simi:.2f}>{self.conf['relevant_wake']}"
                        break

        # 4. 答疑唤醒：检测到提问意图
        if not wake and self.conf["ask_wake"]:
            if self.sent.ask(msg) > self.conf["ask_wake"]:
                wake = True
                reason = "答疑唤醒"

        # 5. 无聊唤醒：检测到无聊/寻求陪伴的意图
        if not wake and self.conf["bored_wake"]:
            if self.sent.bored(msg) > self.conf["bored_wake"]:
                wake = True
                reason = "无聊唤醒"

        # 6. 概率唤醒：随机唤醒
        if not wake and self.conf["prob_wake"]:
            if random.random() < self.conf["prob_wake"]:
                wake = True
                reason = "概率唤醒"

        # 违禁词检查
        if self.conf["wake_forbidden_words"]:
            for word in self.conf["wake_forbidden_words"]:
                if not event.is_admin() and word in event.message_str:
                    logger.debug(f"用户({uid})消息含违禁词：{word}")
                    event.stop_event()
                    return

        # --- 标记唤醒状态 ---
        
        if wake:
            event.is_at_or_wake_command = True
            logger.info(f"群({gid})用户({uid}) {reason}：{msg[:50]}")
            # 注意: 所有检测和防护都在第二层(on_llm_request)处理

    # ==================== 消息合并处理 ====================
    
    async def _handle_message_merge(
        self,
        event: AstrMessageEvent,
        gid: str,
        uid: str,
        member: MemberState,
        now: float
    ):
        """消息合并处理"""
        message_buffer = [event.message_str]
        first_event = event
        
        @session_waiter(timeout=self.conf["merge_delay"], record_history_chains=False)
        async def collect_messages(controller: SessionController, ev: AstrMessageEvent):
            """收集后续消息"""
            nonlocal message_buffer
            
            # 只收集同一用户的消息
            if ev.get_sender_id() != uid:
                logger.debug(f"合并：跳过其他用户({ev.get_sender_id()})的消息")
                return
            
            # 只收集同一群组的消息
            if ev.get_group_id() != gid:
                logger.debug(f"合并：跳过其他群组的消息")
                return
            
            # 防止重复处理第一条消息
            if len(message_buffer) == 1 and ev.message_str == message_buffer[0]:
                logger.debug(f"合并：跳过重复的第一条消息")
                controller.keep(timeout=self.conf["merge_delay"], reset_timeout=True)
                return
            
            # 消息数量限制（防止轰炸攻击）
            if len(message_buffer) >= MAX_MERGE_MESSAGES:
                logger.warning(
                    f"合并：用户({uid})消息数量达到上限({MAX_MERGE_MESSAGES})，"
                    f"强制结束合并"
                )
                controller.stop()
                return
            
            request_cd = self.conf.get("request_cd", 0)
            if request_cd > 0:
                time_since_last_request = time.time() - member.last_request
                if time_since_last_request < request_cd:
                    logger.debug(
                        f"合并：消息间隔过短，请求CD阻止"
                        f"({time_since_last_request:.1f}s < {request_cd}s)"
                    )
                    ev.stop_event()
                    return  # 不合并，也不重置超时
            
            # 收集消息
            message_buffer.append(ev.message_str)
            logger.debug(f"合并：收集用户({uid})消息[{len(message_buffer)}]: {ev.message_str[:50]}")
            
            # 阻止这条消息的默认处理（避免重复调用LLM）
            ev.stop_event()
            
            # 重置超时，继续等待
            controller.keep(timeout=self.conf["merge_delay"], reset_timeout=True)
        
        try:
            await collect_messages(event)
            logger.debug(f"合并：会话被停止")
        except TimeoutError:
            # 超时：合并消息并继续处理
            if len(message_buffer) > 1:
                merged_msg = " ".join(message_buffer)  # 用空格连接
                first_event.message_str = merged_msg
                logger.info(
                    f"合并：用户({uid})合并了{len(message_buffer)}条消息 "
                    f"({merged_msg[:100]}...)"
                )
            # 不 stop_event，让合并后的消息继续传递给 LLM

    # ==================== 第二层: LLM请求级别钩子 ====================
    
    @filter.on_llm_request(priority=99)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        【第二层: LLM请求级别 - 深度防护】
        
        在真正发送LLM请求前执行:
        1. 沉默触发检测(辱骂、闭嘴、AI检测)
        2. 防护机制(沉默状态、闭嘴状态、请求CD)
        3. 消息合并(短时间多条消息合并)
        """
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        
        if not gid or not uid:
            return
        
        g: GroupState = StateManager.get_group(gid)
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)
        
        member = g.members[uid]
        now = time.time()
        msg = event.message_str
        
        # 如果用户正在消息合并状态，跳过检查(由 session_waiter 处理)
        if member.in_merging:
            logger.debug(f"用户({uid})处于消息合并状态，等待合并完成")
            return

        # ========== 1. 沉默触发检测 ==========
        
        # 闭嘴机制: 针对整个群组(立即生效)
        if self.conf["shutup"]:
            shut_th = self.sent.shut(msg)
            if shut_th > self.conf["shutup"]:
                silence_sec = shut_th * self.conf["silence_multiple"]
                g.shutup_until = now + silence_sec
                logger.info(f"群({gid})触发闭嘴，沉默{silence_sec:.1f}秒")
                event.stop_event()
                return

        # 辱骂沉默机制: 针对单个用户(下次生效,本次允许bot回怼)
        if self.conf["insult"]:
            insult_th = self.sent.insult(msg)
            if insult_th > self.conf["insult"]:
                silence_sec = insult_th * self.conf["silence_multiple"]
                member.silence_until = now + silence_sec
                logger.info(f"用户({uid})触发辱骂沉默{silence_sec:.1f}秒(下次生效)")
                # 不阻止本次对话，让bot回怼

        # AI检测沉默机制: 针对单个用户(立即生效)
        if self.conf["ai"]:
            ai_th = self.sent.is_ai(msg)
            if ai_th > self.conf["ai"]:
                silence_sec = ai_th * self.conf["silence_multiple"]
                member.silence_until = now + silence_sec
                logger.info(f"用户({uid})触发AI检测沉默{silence_sec:.1f}秒")
                event.stop_event()
                return

        # ========== 2. 防护机制检查 ==========
        
        # 群组闭嘴检查
        if g.shutup_until > now:
            logger.debug(f"群({gid})处于闭嘴状态，阻止LLM请求")
            event.stop_event()
            return

        # 沉默检查
        if not event.is_admin() and member.silence_until > now:
            logger.debug(f"用户({uid})处于沉默状态，阻止LLM请求")
            event.stop_event()
            return

        # 请求CD检查: 防止消息轰炸
        request_cd_value = self.conf.get("request_cd", 0)
        if request_cd_value > 0:
            time_since_last_request = now - member.last_request
            if time_since_last_request < request_cd_value:
                logger.debug(
                    f"用户({uid})处于请求CD中"
                    f"({time_since_last_request:.1f}s < {request_cd_value}s)"
                )
                event.stop_event()
                return
        
        # 记录请求时间
        member.last_request = now
        
        # ========== 3. 消息合并处理 ==========
        
        # 消息合并: 等待短时间内的后续消息
        if self.conf["merge_delay"] and self.conf["merge_delay"] > 0:
            if not member.in_merging:
                member.in_merging = True
                try:
                    await self._handle_message_merge(event, gid, uid, member, now)
                finally:
                    member.in_merging = False
            else:
                logger.debug(f"用户({uid})已在消息合并状态，跳过")
    
    # ==================== 事件钩子 ====================
    
    @filter.on_llm_response(priority=20)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """LLM响应后的钩子"""
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        
        if not gid or not uid:
            return
            
        g: GroupState = StateManager.get_group(gid)
        member = g.members.get(uid)
        
        if not member:
            return
        
        member.last_response = time.time()
        logger.debug(f"LLM响应完成，更新用户({uid})的last_response时间")

    # ==================== 辅助方法 ====================
    
    async def _get_history_msg(
        self,
        event: AstrMessageEvent,
        role: str = "assistant",
        count: int | None = 0
    ) -> list | None:
        """获取历史消息"""
        try:
            umo = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            
            if not curr_cid:
                return None

            conversation = await self.context.conversation_manager.get_conversation(umo, curr_cid)
            
            if not conversation:
                return None

            history = json.loads(conversation.history or "[]")
            contexts = [
                record["content"]
                for record in history
                if record.get("role") == role and record.get("content")
            ]
            
            return contexts[-count:] if count else contexts

        except Exception as e:
            logger.error(f"获取历史消息失败：{e}")
            return None

    async def _get_llm_respond(
        self,
        event: AstrMessageEvent,
        prompt_template: str
    ) -> str | None:
        """调用 LLM 获取回复"""
        try:
            umo = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            conversation = await self.context.conversation_manager.get_conversation(umo, curr_cid)
            contexts = json.loads(conversation.history)

            personality = self.context.get_using_provider().curr_personality
            personality_prompt = personality["prompt"] if personality else ""

            format_prompt = prompt_template.format(username=event.get_sender_name())

            llm_response = await self.context.get_using_provider().text_chat(
                prompt=format_prompt,
                system_prompt=personality_prompt,
                contexts=contexts,
            )
            
            return llm_response.completion_text

        except Exception as e:
            logger.error(f"LLM调用失败：{e}")
            return None
