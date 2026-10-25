from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

def run_evaluation(test_questions: list[str], ground_truths: list[str], retriever):
    answers = []
    contexts = []
    
    for q in test_questions:
        result = answer_query(q, retriever)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    scores = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
    )
    return scores.to_pandas()