from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class EnterpriseTopology:
    n_nodes: int = 10

    # zone ids
    Z_DMZ = 0
    Z_USER = 1
    Z_DATA = 2
    Z_ADMIN = 3
    Z_SEC = 4

    # node types
    T_WEB = 0
    T_APP = 1
    T_WORKSTATION = 2
    T_DB = 3
    T_FILE = 4
    T_DC = 5
    T_ADMIN_PC = 6
    T_SIEM = 7

    nodes: List[Dict] = None
    edges: List[Tuple[int, int]] = None

    def __post_init__(self):
        # 10 nodes
        self.nodes = [
            {"id": 0, "name": "dmz_web_1", "zone": self.Z_DMZ, "type": self.T_WEB, "criticality": 0.6},
            {"id": 1, "name": "dmz_app_1", "zone": self.Z_DMZ, "type": self.T_APP, "criticality": 0.6},
            {"id": 2, "name": "user_pc_1", "zone": self.Z_USER, "type": self.T_WORKSTATION, "criticality": 0.4},
            {"id": 3, "name": "user_pc_2", "zone": self.Z_USER, "type": self.T_WORKSTATION, "criticality": 0.4},
            {"id": 4, "name": "user_pc_3", "zone": self.Z_USER, "type": self.T_WORKSTATION, "criticality": 0.4},
            {"id": 5, "name": "db_1", "zone": self.Z_DATA, "type": self.T_DB, "criticality": 1.0},
            {"id": 6, "name": "file_1", "zone": self.Z_DATA, "type": self.T_FILE, "criticality": 0.8},
            {"id": 7, "name": "dc_1", "zone": self.Z_ADMIN, "type": self.T_DC, "criticality": 1.0},
            {"id": 8, "name": "admin_pc", "zone": self.Z_ADMIN, "type": self.T_ADMIN_PC, "criticality": 0.7},
            {"id": 9, "name": "siem", "zone": self.Z_SEC, "type": self.T_SIEM, "criticality": 0.5},
        ]

        # undirected edges
        self.edges = [
            (0, 1), (0, 2), (1, 2),
            (2, 3), (3, 4),
            (3, 5), (4, 6),
            (5, 6), (5, 7),
            (7, 8),
            (9, 0), (9, 2), (9, 5),
        ]

    def edge_list(self):
        return self.edges