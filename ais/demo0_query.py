"""
AIS 資料庫查詢 Demo

示範如何查詢 AIS 假資料庫。
"""

from ais_database import (
    get_all_ships,
    get_ship_by_id,
    get_ships_by_type,
    get_known_types
)


if __name__ == "__main__":
    print("=== AIS 資料庫查詢 Demo ===\n")

    # 1. 取得所有船舶
    all_ships = get_all_ships()
    print(f"[1] AIS 資料庫共有 {len(all_ships)} 艘船\n")

    # 2. 列出所有已知類型
    known_types = get_known_types()
    print(f"[2] 已知船舶類型：{known_types}\n")

    # 3. 查詢特定類型
    print("[3] 查詢 Fishing Boat：")
    fishing_boats = get_ships_by_type("Fishing Boat")
    for boat in fishing_boats:
        print(f"    - {boat['name']} (MMSI: {boat['mmsi']})")
    print()

    # 4. 用 ID 查詢
    print("[4] 查詢 ship_002：")
    ship = get_ship_by_id("ship_002")
    if ship:
        print(f"    名稱：{ship['name']}")
        print(f"    類型：{ship['type']}")
        print(f"    船籍：{ship['flag']}")
    print()

    # 5. 列出所有船舶
    print("[5] 所有船舶清單：")
    for ship in all_ships:
        print(f"    - {ship['name']:15} | {ship['type']:15} | {ship['flag']}")
