# similarity.py
import math
import re
from collections import defaultdict, deque

import jieba


class Similarity:
    """
    最终稳定版话题相关性检测
    - 无漂移
    - 无相似度通胀
    - 群号隔离
    """

    def __init__(self,
                 history_limit: int = 120,
                 stopwords=None,
                 bot_template_threshold: int = 2):
        """
        :param history_limit: 每个群最大历史窗口大小
        :param stopwords: 自定义停用词列表
        :param bot_template_threshold: bot 固定句子长度小于N时弱化权重
        """
        self._GROUP_DATA = defaultdict(lambda: {
            "history": deque(maxlen=history_limit),
            "idf": defaultdict(int),
            "total_docs": 0,
        })

        self.stopwords = stopwords or {
            "的", "了", "吗", "吧", "啊", "哦", "嗯", "恩", "你", "我",
            "他", "她", "它", "这", "那", "就", "都", "又"
        }
        self.bot_template_threshold = bot_template_threshold

    # -------------------------
    # 内部工具
    # -------------------------
    def _tokenize(self, text: str):
        text = re.sub(r"[^\w\u4e00-\u9fa5]", " ", text)
        words = jieba.lcut(text)
        return [w for w in words if w not in self.stopwords and len(w.strip()) > 0]

    def _update_idf(self, group_id: str, tokens: set):
        """更新 IDF 信息"""
        data = self._GROUP_DATA[group_id]
        for t in tokens:
            data["idf"][t] += 1 # type: ignore
        data["total_docs"] += 1  # type: ignore

    def _tfidf_vector(self, group_id: str, tokens: list):
        """构建稳定的 TF-IDF 向量"""
        data = self._GROUP_DATA[group_id]
        total_docs = data["total_docs"] or 1

        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        vec = {}
        for t, c in tf.items():
            idf = math.log((total_docs + 1) / (data["idf"][t] + 1)) + 1  # type: ignore
            vec[t] = c * idf
        return vec

    def _cosine(self, v1, v2):
        if not v1 or not v2:
            return 0.0

        # 分子
        dot = 0
        for k, v in v1.items():
            if k in v2:
                dot += v * v2[k]

        # 分母
        norm1 = math.sqrt(sum(v * v for v in v1.values()))
        norm2 = math.sqrt(sum(v * v for v in v2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    # -------------------------
    # 主对外方法
    # -------------------------
    def similarity(self, group_id: str, user_msg: str, bot_msgs: list[str]):
        """
        对用户消息与 bot 最近N条消息做相似度检测
        """
        # 分词
        user_tokens = self._tokenize(user_msg)
        if not user_tokens:
            return 0.0

        # 更新群历史语料
        history_entry = " ".join(user_tokens)
        if history_entry:
            self._GROUP_DATA[group_id]["history"].append(history_entry)  # type: ignore
            self._update_idf(group_id, set(user_tokens))

        # 构建用户向量
        user_vec = self._tfidf_vector(group_id, user_tokens)

        # 对 bot 每条消息做相似度计算
        scores = []
        for bm in bot_msgs:
            if not bm:
                continue

            # 去除模板句子影响（如 “好的，我来了”）
            if len(bm) <= self.bot_template_threshold:
                continue

            bm_tokens = self._tokenize(bm)
            if not bm_tokens:
                continue

            bm_vec = self._tfidf_vector(group_id, bm_tokens)
            sim = self._cosine(user_vec, bm_vec)
            scores.append(sim)

        if not scores:
            return 0.0

        return max(scores)
