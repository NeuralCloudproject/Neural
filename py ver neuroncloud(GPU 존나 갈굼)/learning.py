import sqlite3

class NeuronLearner:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def reward_neuron(self, neuron_id: int, reward: float):
        self.cursor.execute("""
            UPDATE neurons SET reward = reward + ? WHERE id = ?
        """, (reward, neuron_id))
        self.conn.commit()

    def close(self):
        self.conn.close()
