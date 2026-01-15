"""
AIS 假資料庫

Workshop 解答：建立至少 3 艘不同類型的船舶資料
"""

ais_database = {
    "ship_001": {
        "mmsi": "123456789",
        "name": "FISHING STAR",
        "type": "Fishing Boat",
        "flag": "TW",
        "length": 25.5,
        "position": {"lat": 25.0330, "lon": 121.5654},
        "speed": 8.5,
        "course": 180
    },
    "ship_002": {
        "mmsi": "987654321",
        "name": "OCEAN TRADER",
        "type": "Container Ship",
        "flag": "PA",
        "length": 180.0,
        "position": {"lat": 25.1000, "lon": 121.6000},
        "speed": 12.0,
        "course": 270
    },
    "ship_003": {
        "mmsi": "555666777",
        "name": "ISLAND FERRY",
        "type": "Ferry",
        "flag": "TW",
        "length": 85.0,
        "position": {"lat": 25.0500, "lon": 121.5800},
        "speed": 15.0,
        "course": 90
    },
    "ship_004": {
        "mmsi": "111222333",
        "name": "BULK STAR",
        "type": "Bulk Carrier",
        "flag": "LR",
        "length": 200.0,
        "position": {"lat": 25.0800, "lon": 121.6200},
        "speed": 10.0,
        "course": 45
    },
    "ship_005": {
        "mmsi": "444555666",
        "name": "ORE KING",
        "type": "Ore Carrier",
        "flag": "HK",
        "length": 250.0,
        "position": {"lat": 25.1200, "lon": 121.5500},
        "speed": 11.0,
        "course": 135
    }
}


def get_all_ships() -> list[dict]:
    """取得所有 AIS 船舶資料"""
    return [
        {**ship, "ship_id": ship_id}
        for ship_id, ship in ais_database.items()
    ]


def get_ship_by_id(ship_id: str) -> dict | None:
    """用 ID 查詢船舶"""
    ship = ais_database.get(ship_id)
    if ship:
        return {**ship, "ship_id": ship_id}
    return None


def get_ships_by_type(ship_type: str) -> list[dict]:
    """用類型查詢船舶"""
    return [
        {**ship, "ship_id": ship_id}
        for ship_id, ship in ais_database.items()
        if ship["type"] == ship_type
    ]


def get_known_types() -> list[str]:
    """取得所有已知船舶類型"""
    return list(set(ship["type"] for ship in ais_database.values()))


if __name__ == "__main__":
    print("=== AIS 假資料庫 ===\n")
    print(f"已登記船舶數量：{len(ais_database)}")
    print(f"已知類型：{get_known_types()}")
    print()
    for ship in get_all_ships():
        print(f"  - {ship['name']:15} | {ship['type']:15} | {ship['flag']}")
