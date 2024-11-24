from database_manager import DatabaseManager
from visualization import NetworkVisualizer
from learning import NeuronLearner
from gpt_integration import GPTIntegrator

def main():
    db_path = "neuron_network_large.db"
    api_key = "your-openai-api-key"

    # 데이터베이스 초기화 및 뉴런 생성
    db_manager = DatabaseManager(db_path)

    # 뉴런 수와 연결 수 설정
    num_neurons = 1000000
    connections_per_neuron = 1000

    db_manager.create_neurons(num_neurons=num_neurons, connections_per_neuron=connections_per_neuron)

    visualizer = NetworkVisualizer(db_path)
    learner = NeuronLearner(db_path)
    gpt = GPTIntegrator(api_key)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # 랜덤 뉴런 선택
        neuron_id = 0  # 테스트용
        visualizer.visualize_neuron(neuron_id)

        neuron_state = f"Neuron-{neuron_id} (active)"
        response = gpt.chat(neuron_state, user_input)
        print("AI:", response)

        # 학습
        feedback = float(input("Provide feedback (-1.0 to 1.0): "))
        learner.reward_neuron(neuron_id, feedback)

    # 종료
    db_manager.close()
    learner.close()

if __name__ == "__main__":
    main()
