import sqlite3
import random

class DatabaseManager:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_db()

    def _initialize_db(self):
        # 뉴런 테이블 생성
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS neurons (
                id INTEGER PRIMARY KEY,
                threshold REAL,
                rest_potential REAL,
                reward REAL DEFAULT 0
            )
        """)
        # 시냅스 테이블 생성
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS synapses (
                source_id INTEGER,
                target_id INTEGER,
                weight REAL,
                PRIMARY KEY (source_id, target_id)
            )
        """)
        self.conn.commit()

    def create_neurons(self, num_neurons: int, connections_per_neuron: int):
        print(f"Creating {num_neurons} neurons and their connections...")

        # 뉴런 데이터 배치 생성
        neurons = [
            (i, random.uniform(-70, -50), random.uniform(-90, -70))
            for i in range(num_neurons)
        ]
        self.cursor.executemany("""
            INSERT OR IGNORE INTO neurons (id, threshold, rest_potential) 
            VALUES (?, ?, ?)
        """, neurons)
        self.conn.commit()

        # 시냅스 데이터 배치 생성
        for i in range(num_neurons):
            connections = [
                (i, random.randint(0, num_neurons - 1), random.uniform(0.1, 1.0))
                for _ in range(connections_per_neuron)
            ]
            self.cursor.executemany("""
                INSERT OR IGNORE INTO synapses (source_id, target_id, weight) 
                VALUES (?, ?, ?)
            """, connections)

            # 10,000 단위로 진행 상황 출력
            if i % 10000 == 0:
                print(f"Processed {i}/{num_neurons} neurons...")
                self.conn.commit()

        self.conn.commit()
        print("Neuron and synapse creation complete!")

    def close(self):
        self.conn.close()
