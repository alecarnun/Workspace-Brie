import json
from os import makedirs
import os
import pandas as pd
import torchmetrics
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src import figures
from src.models.losses import UserwiseAUCROC
import sacrebleu

def compute_sentence_bleu(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return 0.0

    bleu = sacrebleu.sentence_bleu(candidate, [reference])
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

    # Data for the percentile figures
    percentile_figure_data = {"city": datamodule.city, "metrics": []}
    recall_figure_data = {"city": datamodule.city, "metrics": []}
    ndcg_figure_data = {"city": datamodule.city, "metrics": []}
    # =========================================================
    # LIMIT GLOBAL DEBUG (IMPORTANT)
    # =========================================================
    debug_max_testcases = 200

    for model in model_preds:
        print("=" * 50)
        print(model)
        print("=" * 50)

        test_set = datamodule.test_dataset.dataframe
        test_set["pred"] = model_preds[model]

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

        print("\n--- SUMMARY EVALUATION (LIMITED DEBUG) ---\n")

        summaries_data = []
        top_n = 10

        grouped = list(test_set.groupby("id_test", sort=False))[:debug_max_testcases]

        for id_test, group in grouped:
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
           summaries_data.append({
               "id_test": id_test,
               "summary": summary,
               "reference": reference,
               "top_reviews": " ||| ".join(top_reviews)
           })

           print(f"Generated summaries: {len(summaries_data)}")
           # =========================================================
           # SAVE TO FILE
           # =========================================================

           output_path = f"docs/{datamodule.city}/summaries_{model}.csv"

           df_summaries = pd.DataFrame(summaries_data)
           df_summaries.to_csv(output_path, index=False)

           print(f"Summaries saved to: {output_path}")

        # =========================================================
        # RESTO DEL PIPELINE (UNCHANGED)
        # =========================================================

        train_set = datamodule.train_dataset.dataframe

        # Get no. of images in each test case: not the same unique restaurant
        # images, as a user may have >1 images per restaurant and each test case
        # only has one of them
        images_per_testcase = (
            test_set.groupby("id_test")
            .size()
            .reset_index(name="testcase_num_images")
        )

        # Compute number of photos in each user's train set
        train_photos_per_user = (
            train_set[train_set["take"] == 1]
            .drop_duplicates(keep="first")
            .groupby("id_user")
            .size()
            .reset_index(name="author_num_train_photos")
        )

        # # Compute the percentile metric of each test case
        test_cases = (
            test_set.groupby("id_test").apply(get_testcase_rankingmetrics).reset_index()
        )

        # Add the user and subreddit information
        test_cases = pd.merge(
            test_cases,
            train_photos_per_user,
            left_on="id_user",
            right_on="id_user",
            how="inner",
        )
        test_cases = pd.merge(
            test_cases,
            images_per_testcase,
            left_on="id_test",
            right_on="id_test",
            how="inner",
        )

        # Initialize figure data
        model_percentile_metrics = {
            "min_photos": [],
            "num_test_cases": [],
            "median_percentile": [],
            "model_name": model,
        }

        preds = torch.tensor(test_set["pred"], dtype=torch.float)
        target = torch.tensor(test_set["is_dev"], dtype=torch.long)
        indexes = torch.tensor(test_set["id_test"], dtype=torch.long)
        model_userwise_auroc = UserwiseAUCROC()(
            indexes=indexes, target=target, preds=preds
        )
        print("")
        print(f"AUC (all users, all test cases): {model_userwise_auroc:.3f}")
        print("")

        # Load file numfactors_results.json with json package if it exists, otherwise create it
        if model != "RANDOM" and model != "CNT":
            try:
                with open(f"results/numfactors_results.json", "r") as f:
                    numfactors_results = json.load(f)
            except FileNotFoundError:
                if not os.path.exists("results"):
                    os.makedirs("results")
                numfactors_results = {}

            # Save the auroc for this city, model and num_factors
            if args.city not in numfactors_results:
                numfactors_results[args.city] = {}
            if args.model[0] not in numfactors_results[args.city]:
                numfactors_results[args.city][args.model[0]] = {}
            if str(args.d) not in numfactors_results[args.city][args.model[0]]:
                print(f"Saving {args.city}, {args.model[0]}, {args.d}")
                numfactors_results[args.city][args.model[0]].update(
                    {args.d: float(model_userwise_auroc)}
                )
            else:
                print(
                    f"Already exists: {args.city}, {args.model[0]}, {args.d}, AUROC: {numfactors_results[args.city][args.model[0]][str(args.d)]}"
                )

            # Save the results
            with open(f"results/numfactors_results.json", "w") as f:
                json.dump(numfactors_results, f)

        # We only take into account restaurants with >10 photos
        test_cases = test_cases[test_cases["testcase_num_images"] >= 10]

        # Compute percentile figure metrics
        print(f"Min. imgs  Percentile  Test Cases")
        for i in range(1, 101):
            percentiles = test_cases[test_cases["author_num_train_photos"] >= i][
                "percentile"
            ]

            model_percentile_metrics["min_photos"].append(i)
            model_percentile_metrics["num_test_cases"].append(len(percentiles))
            model_percentile_metrics["median_percentile"].append(percentiles.median())

        #     print(f"{i:<11}{percentiles.median():<12.3f}({len(percentiles)})")
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

        print("\n--- BLEU evaluation (top-N vs user review) ---")

        bleu_scores = []

        top_n = 10  # puedes cambiarlo

        for id_test, group in test_set.groupby("id_test", sort=False):
            group = group.sort_values("pred", ascending=False)

            # review real del usuario
            pos_row = group[group["is_dev"] == 1]

            if len(pos_row) == 0:
                continue

            reference = pos_row.iloc[0]["review_full"]

            # quitar la positiva del ranking
            group_no_pos = group[group["is_dev"] == 0]

            # coger top-N negativas
            top_candidates = group_no_pos.head(top_n)["review_full"].tolist()

            # calcular BLEU para cada una
            bleu_per_case = []

            for candidate in top_candidates:
                bleu = compute_sentence_bleu(reference, candidate)
                bleu_per_case.append(bleu)

            # media por test case
            if len(bleu_per_case) > 0:
                bleu_scores.append(sum(bleu_per_case) / len(bleu_per_case))

        # media global
        if len(bleu_scores) > 0:
            mean_bleu = sum(bleu_scores) / len(bleu_scores)
        else:
            mean_bleu = 0.0

        print(f"Mean BLEU (top-{top_n} vs user review): {mean_bleu:.3f}")
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
