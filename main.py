from retrieval.stage1 import stage1
from retrieval.stage2 import stage2
from retrieval.stage3 import stage3

if __name__ == "__main__":
    user_query = "What is happening in Nepal Right Now?"

    print("\n=== Stage 1: Live Web Retrieval ===")
    stage1_output = stage1(user_query)

    print("\n=== Stage 2: Archival Retrieval ===")
    stage2_output = stage2(user_query, top_k=2)

    print("\n=== Stage 3: Final Answer ===")
    stage3_output = stage3(user_query, stage1_output, stage2_output)

    print("\nFINAL ANSWER:\n", stage3_output)
