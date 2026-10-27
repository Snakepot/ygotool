"""YGOPro 位运算字段解码（教学补充）。

ygocdb 的 `data` 字段是 YGOPro 风格的位掩码整数。注意编码来源：
- **Attribute / Type**：与标准 YGOPro `constant.lua` 一致。
- **Race**：ygocdb 自有顺序，**不等于**标准 YGOPro（标准里龙族=0x80000，此处=0x2000）。
  下表的 RACE 值是 2026-09-14 从全量 cards.zip（14295 张卡）反推得到的权威映射，
  按 `text.types` 中的中文种族名与 `data.race` 一一对应得出。

⚠️ 教学用途：站点返回的 `text.types` 已是可读字符串（如 "[怪兽|通常] 龙/光"），
   实际显示可直接用它；本模块演示"如何从位掩码解码"，便于理解 YGOPro 数据结构。
"""
from __future__ import annotations
from typing import Iterable

# ---- Attribute（标准 1<<n）----
ATTRIBUTE: dict[int, tuple[str, str]] = {
    0x0001: ("地", "Earth"),
    0x0002: ("水", "Water"),
    0x0004: ("炎", "Fire"),
    0x0008: ("风", "Wind"),
    0x0010: ("光", "Light"),
    0x0020: ("暗", "Dark"),
    0x0040: ("神", "Divine"),
}

# ---- Race（ygocdb 自有顺序，1<<n；从全量卡池反推）----
RACE: dict[int, tuple[str, str]] = {
    0x0000001: ("战士", "Warrior"),
    0x0000002: ("魔法师", "Spellcaster"),
    0x0000004: ("天使", "Fairy"),
    0x0000008: ("恶魔", "Fiend"),
    0x0000010: ("不死", "Zombie"),
    0x0000020: ("机械", "Machine"),
    0x0000040: ("水生", "Aqua"),
    0x0000080: ("炎", "Pyro"),
    0x0000100: ("岩石", "Rock"),
    0x0000200: ("鸟兽", "WingedBeast"),
    0x0000400: ("植物", "Plant"),
    0x0000800: ("昆虫", "Insect"),
    0x0001000: ("雷", "Thunder"),
    0x0002000: ("龙", "Dragon"),
    0x0004000: ("兽", "Beast"),
    0x0008000: ("兽战士", "BeastWarrior"),
    0x0010000: ("恐龙", "Dinosaur"),
    0x0020000: ("鱼", "Fish"),
    0x0040000: ("海龙", "SeaSerpent"),
    0x0080000: ("爬虫类", "Reptile"),
    0x0100000: ("念动力", "Psychic"),
    0x0200000: ("幻神兽", "DivineBeast"),
    0x0400000: ("创造神", "CreatorGod"),
    0x0800000: ("幻龙", "Wyrm"),
    0x1000000: ("电子界", "Cyberse"),
    0x2000000: ("幻想魔", "Illusion"),
}

# ---- Type（标准 YGOPro TYPE_*）----
TYPE: dict[int, tuple[str, str]] = {
    0x0000001: ("怪兽", "Monster"),
    0x0000002: ("魔法", "Spell"),
    0x0000004: ("陷阱", "Trap"),
    0x0000010: ("通常", "Normal"),
    0x0000020: ("效果", "Effect"),
    0x0000040: ("融合", "Fusion"),
    0x0000080: ("仪式", "Ritual"),
    0x0000200: ("灵魂", "Spirit"),
    0x0000400: ("同盟", "Union"),
    0x0000800: ("二重", "Dual"),
    0x0001000: ("调整", "Tuner"),
    0x0002000: ("同调", "Synchro"),
    0x0010000: ("速攻", "QuickPlay"),
    0x0020000: ("永续", "Continuous"),
    0x0040000: ("装备", "Equip"),
    0x0080000: ("场地", "Field"),
    0x0100000: ("反击", "Counter"),
    0x0200000: ("反转", "Flip"),
    0x0400000: ("卡通", "Toon"),
    0x0800000: ("超量", "Xyz"),
    0x1000000: ("灵摆", "Pendulum"),
    0x2000000: ("特殊召唤", "SpecialSummon"),
    0x4000000: ("连接", "Link"),
}

# Link 标记方向（标准 YGOPro LINK_MARK_*）— `level` 字段在连接怪兽里存的是箭头位掩码
LINK_MARKERS: dict[int, str] = {
    0x001: "↙", 0x002: "↓", 0x004: "↘",
    0x008: "←", 0x020: "→",
    0x040: "↖", 0x080: "↑", 0x100: "↗",
}

TYPE_LINK = 0x4000000
TYPE_PENDULUM = 0x1000000
TYPE_SPELL = 0x0000002
TYPE_TRAP = 0x0000004
TYPE_MONSTER = 0x0000001


def _bits(value: int, table: dict[int, tuple[str, str]]) -> Iterable[tuple[int, tuple[str, str]]]:
    for bit, name in table.items():
        if value & bit:
            yield bit, name


def decode_attribute(value: int, lang: str = "zh") -> list[str]:
    """属性位掩码 -> 名称列表。attr=16 -> ['光']."""
    return [n[0] if lang == "zh" else n[1] for _, n in _bits(value or 0, ATTRIBUTE)]


def decode_race(value: int, lang: str = "zh") -> list[str]:
    """种族位掩码 -> 名称列表。race=8192 -> ['龙']."""
    return [n[0] if lang == "zh" else n[1] for _, n in _bits(value or 0, RACE)]


def decode_type(value: int, lang: str = "zh") -> list[str]:
    """类型位掩码 -> 标签列表。type=0x11 -> ['怪兽','通常']."""
    return [n[0] if lang == "zh" else n[1] for _, n in _bits(value or 0, TYPE)]


def decode_level(level: int, type_: int = 0) -> dict:
    """解码 level 字段。

    - 连接怪兽：level 是 Link 箭头位掩码 -> {'link': N, 'markers': ['↙','↓',...]}
    - 灵摆怪兽：(lscale<<24)|(rscale<<16)|level -> {'level': n, 'lscale': a, 'rscale': b}
    - 其它怪兽/魔法陷阱：level 即为星级（魔法陷阱为 0）
    """
    lv = level or 0
    if type_ & TYPE_LINK:
        # ygocdb 的 data.level 存的是 Link 数值（LINK-2→2、LINK-3→3…），不是箭头位掩码。
        # 箭头方向（↙↘等）仅编码在 text.types 字符串里，无法从 data 解出。
        return {"link": lv, "markers": []}
    if type_ & TYPE_PENDULUM:
        lscale = (lv >> 24) & 0xFF
        rscale = (lv >> 16) & 0xFF
        star = lv & 0xFFFF
        return {"level": star, "lscale": lscale, "rscale": rscale}
    return {"level": lv}


def decode_setcode(setcode: int) -> list[int]:
    """解码 setcode（系列/字段）。

    YGOPro 的 setcode 是 16-bit 分组打包：每个 16-bit 段为 (count<<12)|index。
    返回所有 index（如 0xDD=221 -> [221]，即「青眼」）。索引→系列名需单独表，超出本模块范围。
    """
    out = []
    v = setcode or 0
    while v:
        seg = v & 0xFFFF
        idx = seg & 0x0FFF
        if idx:
            out.append(idx)
        v >>= 16
    return out


def decode_card(data: dict, lang: str = "zh") -> dict:
    """把一张卡的 `data` 字典整体解码为人类可读结构。"""
    type_ = data.get("type", 0)
    return {
        "attribute": decode_attribute(data.get("attribute", 0), lang),
        "race": decode_race(data.get("race", 0), lang),
        "type": decode_type(type_, lang),
        "level": decode_level(data.get("level", 0), type_),
        "atk": data.get("atk"),
        "def": data.get("def"),
        "setcode": decode_setcode(data.get("setcode", 0)),
        "ot": data.get("ot"),
    }
