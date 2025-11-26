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
from .sentiment import Sentiment
from .similarity import Similarity

# 内置指令文本
BUILT_CMDS = [
    "llm",
    "t2i",
    "tts",
    "sid",
    "op",
    "wl",
    "dashboard_update",
    "alter_cmd",
    "provider",
    "model",
    "plugin",
    "plugin ls",
    "new",
    "switch",
    "rename",
    "del",
    "reset",
    "history",
    "persona",
    "tool ls",
    "key",
    "websearch",
]


class MemberState(BaseModel):
    uid: str
    silence_until: float = 0.0  # 沉默到何时
    last_wake: float = 0.0  # 最后唤醒bot的时间
    pend: list[AstrMessageEvent] = Field(default_factory=list)  # 事件缓存
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GroupState(BaseModel):
    gid: str
    members: dict[str, MemberState] = Field(default_factory=dict)
    shutup_until: float = 0.0  # 闭嘴到何时


class StateManager:
    """内存状态管理"""

    _groups: dict[str, GroupState] = {}

    @classmethod
    def get_group(cls, gid: str) -> GroupState:
        if gid not in cls._groups:
            cls._groups[gid] = GroupState(gid=gid)
        return cls._groups[gid]


@register(
    "astrbot_plugin_wakepro",
    "Zhalslar",
    "更强大的唤醒增强插件",
    "v1.1.2",
)
class WakeProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.sent = Sentiment()
        self.commands = self._get_all_commands()


    def _get_all_commands(self) -> list[str]:
        """遍历所有注册的处理器获取所有命令"""
        commands = []
        for handler in star_handlers_registry:
            for fl in handler.event_filters:
                if isinstance(fl, CommandFilter):
                    commands.append(fl.command_name)
                    break
                elif isinstance(fl, CommandGroupFilter):
                    commands.append(fl.group_name)
                    break
        logger.debug(f"插件的指令列表：{commands}")
        return commands

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=99)
    async def on_group_msg(self, event: AstrMessageEvent):
        """主入口"""
        chain = event.get_messages()
        bid: str = event.get_self_id()
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        msg: str = event.message_str
        g: GroupState = StateManager.get_group(gid)

        # 只处理文本
        if not msg:
            return
        cmd = msg.split(" ", 1)[0]

        # 群聊黑白名单 / 用户黑名单
        if uid == bid:
            return
        if self.conf["group_whitelist"] and gid not in self.conf["group_whitelist"]:
            return
        if gid in self.conf["group_blacklist"] and not event.is_admin():
            event.stop_event()
            return
        if uid in self.conf.get("user_blacklist", []):
            event.stop_event()
            return

        # 更新成员状态
        if uid not in g.members:
            g.members[uid] = MemberState(uid=uid)

        member = g.members[uid]
        now = time.time()

        # 唤醒CD检查
        if now - g.members[uid].last_wake < self.conf["wake_cd"]:
            logger.debug(f"{uid} 处于唤醒CD中, 忽略此次唤醒")
            event.stop_event()
            return

        # 唤醒违禁词检查
        if self.conf["wake_forbidden_words"]:
            for word in self.conf["wake_forbidden_words"]:
                if not event.is_admin() and word in event.message_str:
                    logger.debug(f"{uid} 消息中含有唤醒屏蔽词, 忽略此次唤醒")
                    event.stop_event()
                    return

        # 屏蔽内置指令
        if self.conf["block_builtin"]:
            if not event.is_admin() and event.message_str in BUILT_CMDS:
                logger.debug(f"{uid} 触发内置指令, 忽略此次唤醒")
                event.stop_event()
                return

        # 闭嘴检查
        if g.shutup_until > now:
            event.stop_event()
            return

        # 沉默检查（辱骂/人机）
        if not event.is_admin() and member.silence_until > now:
            event.stop_event()
            return

        # 消息缓存与合并
        if cmd not in self.commands:
            event.set_extra("orig_message", event.message_str)
            event.set_extra("timestamp", now)
            async with member.lock:
                if (
                    member.pend
                    and now - member.pend[-1].get_extra("timestamp")  # type: ignore
                    < self.conf["pend_cd"]
                ):
                    msgs: list[str] = [
                        e.get_extra("orig_message") or "" for e in member.pend
                    ]  # type: ignore
                    for e in member.pend:
                        e.stop_event()
                    event.message_str = "。".join(msgs + [event.message_str])
                    logger.debug(f"已合并{len(member.pend)}条缓存消息：{event.message_str}")

        # 空@回复
        if (
            not msg
            and len(chain) == 1
            and isinstance(chain[0], At)
            and str(chain[0].qq) == bid
        ):
            if text := await self._get_llm_respond(
                event, self.conf["empty_mention_pt"]
            ):
                await event.send(event.plain_result(text))
                event.stop_event()
                return

        # 各类唤醒条件
        wake = event.is_at_or_wake_command
        reason = "at_or_cmd"

        # 提及唤醒
        if not wake and self.conf["mention_wake"]:
            names = [n for n in self.conf["mention_wake"] if n]
            for n in names:
                if n and n in msg:
                    wake = True
                    reason = f"提及唤醒({n})"
                    break

        # 唤醒延长（如果已经处于唤醒状态且在 wake_extend 秒内，每个用户单独延长唤醒时间）
        if (
            not wake
            and self.conf["wake_extend"]
            and (now - member.last_wake) <= int(self.conf["wake_extend"] or 0)
        ):
            wake = True
            reason = "唤醒延长"

        # 话题相关性唤醒
        if not wake and self.conf["relevant_wake"]:
            if bmsgs := await self._get_history_msg(event, count=5):
                for bmsg in bmsgs:
                    simi = Similarity.cosine(msg, bmsg, gid)
                    if simi > self.conf["relevant_wake"]:
                        wake = True
                        reason = f"话题相关性{simi}>{self.conf['relevant_wake']}"
                        break

        # 答疑唤醒
        if (
            not wake
            and self.conf["ask_wake"]
            and self.sent.ask(msg) > self.conf["ask_wake"]
        ):
            wake = True
            reason = "答疑唤醒"

        # 无聊唤醒
        if (
            not wake
            and self.conf["bored_wake"]
            and self.sent.bored(msg) > self.conf["bored_wake"]
        ):
            wake = True
            reason = "无聊唤醒"

        # 概率唤醒
        if (
            not wake
            and self.conf["prob_wake"]
            and random.random() < self.conf["prob_wake"]
        ):
            wake = True
            reason = "概率唤醒"

        # 触发唤醒
        if wake:
            # 缓存消息
            if cmd not in self.commands:
                member.pend.append(event)
                logger.debug(f"已添加event到缓存：{len(member.pend)}")
            # # 记录唤醒时间
            member.last_wake = now
            # 触发唤醒
            event.is_at_or_wake_command = True
            # 记录日志
            logger.info(f"[wakepro] 群({gid}){reason}：{msg}")

        # 闭嘴机制(对当前群聊闭嘴)
        if self.conf["shutup"]:
            shut_th = self.sent.shut(msg)
            if shut_th > self.conf["shutup"]:
                silence_sec = shut_th * self.conf["silence_multiple"]
                g.shutup_until = now + silence_sec
                reason = f"闭嘴沉默{silence_sec}秒"
                logger.info(f"[wakepro] 群({gid}){reason}：{msg}")
                event.stop_event()
                return

        # 辱骂沉默机制(对单个用户沉默)
        if self.conf["insult"]:
            insult_th = self.sent.insult(msg)
            if insult_th > self.conf["insult"]:
                silence_sec = insult_th * self.conf["silence_multiple"]
                member.silence_until = now + silence_sec
                reason = f"辱骂沉默{silence_sec}秒"
                logger.info(f"[wakepro] 群({gid})用户({uid}){reason}：{msg}")
                # event.stop_event() 本轮对话不沉默，方便回怼
                return

        # AI沉默机制(对单个用户沉默)
        if self.conf["ai"]:
            ai_th = self.sent.is_ai(msg)
            if ai_th > self.conf["ai"]:
                silence_sec = ai_th * self.conf["silence_multiple"]
                member.silence_until = now + silence_sec
                reason = f"AI沉默{silence_sec}秒"
                logger.info(f"[wakepro] 群({gid})用户({uid}){reason}：{msg}")
                event.stop_event()
                return

    @filter.on_decorating_result(priority=20)
    async def on_message(self, event: AstrMessageEvent):
        """发送消息前，清空请求消息的缓存"""
        gid: str = event.get_group_id()
        uid: str = event.get_sender_id()
        result = event.get_result()
        if not gid or not uid or not result:
            return
        g: GroupState = StateManager.get_group(gid)
        member = g.members.get(uid)
        if not member:
            return
        async with g.members[uid].lock:
            member.pend.clear()

    async def _get_history_msg(
        self, event: AstrMessageEvent, role: str = "assistant", count: int | None = 0
    ) -> list | None:
        """获取历史消息"""
        try:
            umo = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                umo
            )
            if not curr_cid:
                return None

            conversation = await self.context.conversation_manager.get_conversation(
                umo, curr_cid
            )
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
        self, event: AstrMessageEvent, prompt_template: str
    ) -> str | None:
        """调用llm回复"""
        try:
            umo = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                umo
            )
            conversation = await self.context.conversation_manager.get_conversation(
                umo, curr_cid
            )
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
            logger.error(f"LLM 调用失败：{e}")
            return None
