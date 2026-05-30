from os import makedirs
import numpy as np
import pandas as pd
import torchmetrics
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src import figures
from src.models.losses import UserwiseAUCROC
import sacrebleu
from rouge_score import rouge_scorer

def compute_bleu_multi_ref(reference_list, candidate):
    if not reference_list or not isinstance(candidate, str):
        return 0.0

    bleu = sacrebleu.sentence_bleu(candidate, reference_list)
    return bleu.score / 100.0


def get_testcase_rankingmetrics(test_case: pd.DataFrame):
    # Input: model probabilities and targets of a test case
    # Output: percentile of this test case and raw position of the author's image

    sorted_ranking = test_case.sort_values("pred", ascending=False).reset_index(
        drop=True
    )

    dev_position = sorted_ranking["is_dev"].idxmax()

    return pd.DataFrame(
        {
            "dev_position": [dev_position],
            "percentile": [dev_position / len(sorted_ranking)],
            "id_user": sorted_ranking["id_user"][0],
            "id_restaurant": sorted_ranking["id_restaurant"][0],
        }
    )


def test_tripadvisor_authorship_task(datamodule, model_preds, args):
    makedirs("docs/" + datamodule.city, exist_ok=True)
    makedirs("figures/" + datamodule.city, exist_ok=True)

    # Data for the percentile figures
    percentile_figure_data = {"city": datamodule.city, "metrics": []}
    bleu_figure_data = {"city": datamodule.city, "metrics": []}
    recall_figure_data = {"city": datamodule.city, "metrics": []}
    ndcg_figure_data = {"city": datamodule.city, "metrics": []}
    # =========================================================
    # LIMIT GLOBAL DEBUG (IMPORTANT)
    # =========================================================
    debug_max_testcases = 700

    # =========================================================
    # CAMBIO 1 — MODEL ONLY ONCE PER MODEL (OUTSIDE LOOP)
    # =========================================================
    summarizer_model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
        summarizer_model_name
    )
    summarizer_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summarizer_model.to(device)

    for model in model_preds:
        print("=" * 50)
        print(model)
        print("=" * 50)

        test_set = datamodule.test_dataset.dataframe
        test_set["pred"] = model_preds[model]

        print("\n--- SUMMARY EVALUATION (LIMITED DEBUG) ---\n")

        summaries_data = []
        top_n = 10

        for i, (id_test, group) in enumerate(test_set.groupby("id_test", sort=False)):
            if i >= debug_max_testcases:
                break

            group = group.sort_values("pred", ascending=False)

            pos_row = group[group["is_dev"] == 1]
            if len(pos_row) == 0:
                continue

            reference = pos_row.iloc[0]["review_full"]
            top_reviews = group.head(top_n)["review_full"].tolist()

            input_text = "summarize the following reviews:\n\n" + "\n".join(top_reviews)

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)

            with torch.no_grad():
               output = summarizer_model.generate(
                   **inputs,
                   max_length=60,
                   num_beams=2
            )

            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            # =========================
            # MÉTRICAS
            # =========================
            bleu = compute_bleu_multi_ref(top_reviews, summary)

            tokens = summary.split()
            summary_len = len(tokens)
            reference_len = len(reference.split())

            ref_lens = [len(r.split()) for r in top_reviews]
            ref_len_mean = sum(ref_lens) / len(ref_lens) if ref_lens else 0

            length_ratio = summary_len / ref_len_mean if ref_len_mean > 0 else 0

            distinct_1 = len(set(tokens)) / len(tokens) if tokens else 0
            bigrams = list(zip(tokens, tokens[1:]))
            distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

            input_words = set(" ".join(top_reviews).split())
            summary_words = set(tokens)
            coverage = len(summary_words & input_words) / len(summary_words) if summary_words else 0

            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge = scorer.score(reference, summary)["rougeL"].fmeasure

            summaries_data.append({
                "id_test": id_test,
                "summary": summary,
                "reference": reference,
                "summary_len": summary_len,
                "len_reference": reference_len,
                "ref_len_mean": ref_len_mean,
                "length_ratio": length_ratio,
                "bleu": bleu,
                "distinct_1": distinct_1,
                "distinct_2": distinct_2,
                "coverage": coverage,
                "rouge": rouge
            })

            print(f"[{i+1}/{debug_max_testcases}] Generated summaries: {len(summaries_data)}")
            # =========================================================
            # SAVE TO FILE
            # =========================================================

        output_path = f"docs/{datamodule.city}/summaries_{model}.csv"

        df_summaries = pd.DataFrame(summaries_data)
        df_summaries.set_index("id_test", inplace=True)

        df_summaries.to_csv(output_path)
        print(f"Summaries saved to: {output_path}")

        # =========================================================
        # MERGE WITH USER INFO
        # =========================================================

        train_set = datamodule.train_dataset.dataframe

        train_photos_per_user = (
            train_set[train_set["take"] == 1]
            .drop_duplicates()
            .groupby("id_user")
            .size()
            .reset_index(name="author_num_train_photos")
        )

        test_cases = (
            test_set.groupby("id_test")
            .apply(get_testcase_rankingmetrics)
            .reset_index()
        )

        test_cases = pd.merge(
            test_cases,
            train_photos_per_user,
            on="id_user",
            how="inner"
        )

        df_analysis = pd.merge(
            df_summaries.reset_index(),
            test_cases[["id_test", "author_num_train_photos"]],
            on="id_test",
            how="inner"
        )

        bleu_by_photos = {
            "min_photos": [],
            "num_cases": [],
            "mean_bleu": []
        }

        min_support = 20

        for i in range(1, 101):
            subset = df_analysis[df_analysis["author_num_train_photos"] >= i]

            bleu_by_photos["min_photos"].append(i)
            bleu_by_photos["num_cases"].append(len(subset))

            if len(subset) >= min_support:
                bleu_by_photos["mean_bleu"].append(subset["bleu"].mean())
            else:
                bleu_by_photos["mean_bleu"].append(np.nan)

        bleu_figure_data["metrics"].append({
            "model_name": model,
            "min_photos": list(range(1, 101)),
            "mean_bleu": bleu_by_photos["mean_bleu"]
        })

        df_bleu = pd.DataFrame(bleu_by_photos)

        output_bleu = f"docs/{datamodule.city}/bleu_by_photos_{model}.csv"
        df_bleu.to_csv(output_bleu, index=False)

        print(f"BLEU by photos saved to: {output_bleu}")

        # =========================================================
        # PERCENTILE METRICS (ORIGINAL PIPELINE)
        # =========================================================
        images_per_testcase = (
            test_set.groupby("id_test")
            .size()
            .reset_index(name="testcase_num_images")
        )

        test_cases = pd.merge(
            test_cases,
            images_per_testcase,
            on="id_test",
            how="inner"
        )

        test_cases = test_cases[test_cases["testcase_num_images"] >= 10]

        model_percentile_metrics = {
            "min_photos": [],
            "num_test_cases": [],
            "median_percentile": [],
            "model_name": model,
        }

        for i in range(1, 101):
            percentiles = test_cases[
                test_cases["author_num_train_photos"] >= i
                ]["percentile"]

            model_percentile_metrics["min_photos"].append(i)
            model_percentile_metrics["num_test_cases"].append(len(percentiles))
            model_percentile_metrics["median_percentile"].append(percentiles.median())

        percentile_figure_data["metrics"].append(model_percentile_metrics)

       # For the recall metric, only include users with >= train images

        test_cases = test_cases[test_cases["author_num_train_photos"] >= 10]

        # Initialize recall table data
        model_recall_metrics = {"k": [], "Recall@10": [], "model_name": model}
        model_ndcg_metrics = {"k": [], "NDCG@10": [], "model_name": model}

        test_set = test_set[
            test_set["id_test"].isin(test_cases["id_test"])
        ].reset_index(drop=True)

        if test_set.empty:
            print("No test cases after filtering → skipping Recall/NDCG/BLEU")
            continue

        print("")

        preds = torch.tensor(test_set["pred"], dtype=torch.float)
        target = torch.tensor(test_set["is_dev"], dtype=torch.long)
        indexes = torch.tensor(test_set["id_test"], dtype=torch.long)
        # % of test cases where the image was in position k=1,2,3...10 (Recall at k)
        print("k  Recall@10  NDCG@10")
        for k in range(1, 10 + 1):
            recall_k = torchmetrics.RetrievalRecall(top_k=k)(
                preds=preds, target=target, indexes=indexes
            )
            model_recall_metrics["k"].append(k)
            model_recall_metrics["Recall@10"].append(recall_k)

            ndcg_k = torchmetrics.RetrievalNormalizedDCG(top_k=k)(
                preds=preds, target=target, indexes=indexes
            )

            model_ndcg_metrics["k"].append(k)
            model_ndcg_metrics["NDCG@10"].append(ndcg_k)
            print(f"{k:<3}{recall_k:<8.3f}{ndcg_k:.3f}")

        recall_figure_data["metrics"].append(model_recall_metrics)
        ndcg_figure_data["metrics"].append(model_ndcg_metrics)

        model_userwise_auroc = UserwiseAUCROC()(
            indexes=indexes, target=target, preds=preds
        )
        print("")
        print(
            f"AUC (users with >=10 photos, test cases with size >=10): {model_userwise_auroc:.3f}"
        )
        print("")

    # figures.retrieval_figure(recall_figure_data, "Recall@10")
    # figures.retrieval_figure(ndcg_figure_data, "NDCG@10")
    figures.percentile_figure(percentile_figure_data)
    figures.bleu_figure(bleu_figure_data)
