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

from sentence_transformers import SentenceTransformer
import torch.nn.functional as f

def compute_bleu_multi_ref(reference_list, candidate):
    if not reference_list or not isinstance(candidate, str):
        return 0.0

    reference_list = [
        str(r) for r in reference_list
        if r is not None and str(r).strip() != ""
    ]

    if len(reference_list) == 0:
        return 0.0

    bleu = sacrebleu.BLEU(effective_order=True)

    score = bleu.sentence_score(
        hypothesis=candidate,
        references=reference_list
    )

    return score.score / 100

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

def compute_all_metrics(text, references, rouge_scorer_global):
    # BLEU
    bleu = compute_bleu_multi_ref(references, text)

    # ROUGE
    rouge = max(
        rouge_scorer_global.score(ref, text)["rougeL"].fmeasure
        for ref in references
    ) if references else 0.0

    # TOKENS
    tokens = text.split()
    summary_len = len(tokens)

    ref_lens = [len(r.split()) for r in references]
    ref_len_mean = sum(ref_lens) / len(ref_lens) if ref_lens else 0

    # Relative length
    length_ratio = summary_len / ref_len_mean if ref_len_mean > 0 else 0

    # Diversity
    distinct_1 = len(set(tokens)) / len(tokens) if tokens else 0

    bigrams = list(zip(tokens, tokens[1:]))
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0

    # Coverage
    input_words = set(" ".join(references).split())
    summary_words = set(tokens)

    coverage = len(summary_words & input_words) / len(summary_words) if summary_words else 0

    return {
        "bleu": bleu,
        "rouge": rouge,
        "length_ratio": length_ratio,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "coverage": coverage,
    }

def safe_references(candidate, refs):
    return [r for r in refs if r != candidate]

def test_tripadvisor_authorship_task(datamodule, model_preds, args):
    makedirs("docs/" + datamodule.city, exist_ok=True)
    makedirs("figures/" + datamodule.city, exist_ok=True)

    # Data for the percentile figures
    #percentile_figure_data = {"city": datamodule.city, "metrics": []}
    bleu_figure_data = {"city": datamodule.city, "metrics": []}
    rouge_figure_data = {"city": datamodule.city, "metrics": []}
    dist1_figure_data = {"city": datamodule.city, "metrics": []}
    dist2_figure_data = {"city": datamodule.city, "metrics": []}
    cov_figure_data = {"city": datamodule.city, "metrics": []}
    len_figure_data = {"city": datamodule.city, "metrics": []}
    #recall_figure_data = {"city": datamodule.city, "metrics": []}
    #ndcg_figure_data = {"city": datamodule.city, "metrics": []}

    debug_max_testcases = 2000

    summarizer_model_name = "facebook/BART-large-CNN"

    tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
        summarizer_model_name
    )
    summarizer_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summarizer_model.to(device)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    for model in model_preds:
        if model != "BRIE":
            continue

        print("=" * 50)
        print(model)
        print("=" * 50)

        base_test_set = datamodule.test_dataset.dataframe.copy()

        test_set = base_test_set.copy()
        test_set["pred"] = model_preds[model]

        print("\n--- SUMMARY EVALUATION (LIMITED DEBUG) ---\n")

        # ANÁLISIS DEL DATASET
        print("=== ANÁLISIS DEL DATASET ===")
        test_sizes = []
        is_dev_counts = []

        for id_test, group in base_test_set.groupby("id_test"):
            test_sizes.append(len(group))
            is_dev_counts.append(group["is_dev"].sum())

        print(f"Promedio de reviews por id_test: {np.mean(test_sizes):.2f}")
        print(f"Mediana de reviews por id_test: {np.median(test_sizes):.2f}")
        print(f"Promedio de is_dev=1 por id_test: {np.mean(is_dev_counts):.2f}")
        print(f"Max reviews en un id_test: {max(test_sizes)}")
        print(f"Min reviews en un id_test: {min(test_sizes)}")

        # Verificar que cada id_test tiene exactamente 1 is_dev=1
        casos_incorrectos = sum(1 for c in is_dev_counts if c != 1)
        print(f"Casos con is_dev != 1: {casos_incorrectos}")
        print("=" * 50)
        print()

        summaries_data = []
        top_n = 10

        original_groups = dict(tuple(base_test_set.groupby("id_test")))

        rouge_scorer_global = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        for i, (id_test, group) in enumerate(test_set.groupby("id_test", sort=False)):
            if i >= debug_max_testcases:
                break

            group = group.sort_values("pred", ascending=False)

            pos_row = group[group["is_dev"] == 1]
            if len(pos_row) == 0:
                continue

            # Obtener grupo original (sin ranking de BRIE) para las referencias
            original_group = original_groups[id_test]

            # Referencias para BRIE/CNT/RANDOM: SOLO la review con is_dev=1
            ground_truth = original_group[original_group["is_dev"] == 1]["review_full"].dropna().tolist()

            if len(ground_truth) == 0:
                continue

            # Referencias para BRIE+SUM: TODAS las reviews (porque resume múltiples)
            all_reviews_for_sum = original_group["review_full"].dropna().tolist()
            all_reviews_for_sum = list(dict.fromkeys(all_reviews_for_sum))

            top_reviews = group.head(top_n)["review_full"].tolist()

            # =========================
            # MÉTODOS
            # =========================

            # BRIE (sin summarization): Usar solo el top-1 rankeado por BRIE
            brie_text = group.iloc[0]["review_full"]

            # BRIE + summarization
            input_text = (
                    "You are summarizing restaurant reviews.\n"
                    "Write a concise overall opinion.\n\n"
                    + "\n\n".join(top_reviews[:8])
            )

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

            brie_summary = tokenizer.decode(output[0], skip_special_tokens=True)

            # CNT (baseline independiente del ranking)
            # CNT (centroid-based selection)
            texts = original_group["review_full"].tolist()

            if len(texts) == 0:
                cnt_text = ""
            else:
                embeddings = embedder.encode(texts, convert_to_tensor=True)

                centroid = embeddings.mean(dim=0)

                similarities = f.cosine_similarity(
                    embeddings,
                    centroid.unsqueeze(0),
                    dim=1
                )

                cnt_text = texts[int(torch.argmax(similarities))]

            # RANDOM (también independiente del ranking)
            random_text = original_group.sample(n=1, random_state=int(id_test)).iloc[0]["review_full"]

            if i < 3:
                print("\n--- DEBUG SAMPLE ---")
                print(f"id_test: {id_test}")
                print(f"Original group size: {len(original_group)}")
                print(f"Ground truth (is_dev=1): {ground_truth[0][:80]}...")
                print(f"BRIE ranked is_dev=1 at top? {group.iloc[0]['is_dev'] == 1}")

                # Mostrar ranking completo
                print("\nRANKING de BRIE (top-5):")
                for idx, row in group.head(5).iterrows():
                    marker = " ← TOP-1 (BRIE selects this)" if idx == group.index[0] else ""
                    dev_marker = " [is_dev=1 - GROUND TRUTH]" if row['is_dev'] == 1 else " [is_dev=0]"
                    print(f"  pred={row['pred']:.4f}{dev_marker}: {row['review_full'][:80]}...{marker}")

                print(f"\nBRIE (top-1): {brie_text[:100]}")
                print(f"BRIE+SUM: {brie_summary[:100]}")
                print(f"RANDOM: {random_text[:100]}")
                print(f"CNT: {cnt_text[:100]}")

            # =========================
            # MÉTRICAS
            # =========================
            # BRIE, CNT, RANDOM: Comparar contra ground truth (is_dev=1)
            metrics_brie = compute_all_metrics(
                brie_text,
                ground_truth,
                rouge_scorer_global
            )

            # BRIE+SUM: Comparar contra TODAS las reviews (usa safe_references)
            metrics_brie_sum = compute_all_metrics(
                brie_summary,
                safe_references(brie_summary, all_reviews_for_sum),
                rouge_scorer_global
            )

            metrics_random = compute_all_metrics(
                random_text,
                ground_truth,
                rouge_scorer_global
            )

            metrics_cnt = compute_all_metrics(
                cnt_text,
                ground_truth,
                rouge_scorer_global
            )

            # =========================
            # GUARDAR
            # =========================

            summaries_data.append({
                "id_test": id_test,

                # BLEU
                "bleu_brie": metrics_brie["bleu"],
                "bleu_brie_sum": metrics_brie_sum["bleu"],
                "bleu_random": metrics_random["bleu"],
                "bleu_cnt": metrics_cnt["bleu"],

                # ROUGE
                "rouge_brie": metrics_brie["rouge"],
                "rouge_brie_sum": metrics_brie_sum["rouge"],
                "rouge_random": metrics_random["rouge"],
                "rouge_cnt": metrics_cnt["rouge"],

                # DIVERSITY
                "dist1_brie": metrics_brie["distinct_1"],
                "dist1_brie_sum": metrics_brie_sum["distinct_1"],
                "dist1_random": metrics_random["distinct_1"],
                "dist1_cnt": metrics_cnt["distinct_1"],

                "dist2_brie": metrics_brie["distinct_2"],
                "dist2_brie_sum": metrics_brie_sum["distinct_2"],
                "dist2_random": metrics_random["distinct_2"],
                "dist2_cnt": metrics_cnt["distinct_2"],

                # COVERAGE
                "cov_brie": metrics_brie["coverage"],
                "cov_brie_sum": metrics_brie_sum["coverage"],
                "cov_random": metrics_random["coverage"],
                "cov_cnt": metrics_cnt["coverage"],

                # LENGTH
                "len_brie": metrics_brie["length_ratio"],
                "len_brie_sum": metrics_brie_sum["length_ratio"],
                "len_random": metrics_random["length_ratio"],
                "len_cnt": metrics_cnt["length_ratio"],
            })

            print(f"[{i+1}/{debug_max_testcases}] Generated summaries: {len(summaries_data)}")
            # =========================================================
            # SAVE TO FILE
            # =========================================================

        output_path = f"docs/{datamodule.city}/summaries_{model}.csv"

        df_summaries = pd.DataFrame(summaries_data)
        df_summaries.set_index("id_test", inplace=True)

        print(f"\nGLOBAL METRICS ({model})")

        print(f"BLEU BRIE      : {df_summaries['bleu_brie'].mean():.4f}")
        print(f"BLEU BRIE+SUM  : {df_summaries['bleu_brie_sum'].mean():.4f}")
        print(f"BLEU RANDOM    : {df_summaries['bleu_random'].mean():.4f}")
        print(f"BLEU CNT       : {df_summaries['bleu_cnt'].mean():.4f}")

        print(f"\nROUGE BRIE     : {df_summaries['rouge_brie'].mean():.4f}")
        print(f"ROUGE BRIE+SUM : {df_summaries['rouge_brie_sum'].mean():.4f}")
        print(f"ROUGE RANDOM   : {df_summaries['rouge_random'].mean():.4f}")
        print(f"ROUGE CNT      : {df_summaries['rouge_cnt'].mean():.4f}")

        # ANÁLISIS ADICIONAL: ¿BRIE es mejor cuando acierta?
        print("\n--- ANÁLISIS POR RANKING ---")

        # Añadir columna para saber si BRIE acertó
        test_set_with_rank = test_set.copy()
        test_set_with_rank = test_set_with_rank.sort_values(['id_test', 'pred'], ascending=[True, False])
        test_set_with_rank['rank'] = test_set_with_rank.groupby('id_test').cumcount() + 1

        # Casos donde BRIE pone is_dev=1 en top-1
        top1_correct = test_set_with_rank[(test_set_with_rank['rank'] == 1) & (test_set_with_rank['is_dev'] == 1)]['id_test'].unique()

        if len(top1_correct) > 0:
            df_correct = df_summaries[df_summaries.index.isin(top1_correct)]
            df_incorrect = df_summaries[~df_summaries.index.isin(top1_correct)]

            print(f"\nCasos donde BRIE acierta top-1 (n={len(df_correct)}):")
            print(f"  BLEU BRIE: {df_correct['bleu_brie'].mean():.4f}")
            print(f"  BLEU CNT:  {df_correct['bleu_cnt'].mean():.4f}")
            print(f"  BLEU RAND: {df_correct['bleu_random'].mean():.4f}")

            print(f"\nCasos donde BRIE falla top-1 (n={len(df_incorrect)}):")
            print(f"  BLEU BRIE: {df_incorrect['bleu_brie'].mean():.4f}")
            print(f"  BLEU CNT:  {df_incorrect['bleu_cnt'].mean():.4f}")
            print(f"  BLEU RAND: {df_incorrect['bleu_random'].mean():.4f}")

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

        rows = []

        for i in range(1, 101):
            subset = df_analysis[df_analysis["author_num_train_photos"] >= i]

            rows.append({
                "min_photos": i,
                "num_cases": len(subset),

                # BLEU
                "bleu_brie": subset["bleu_brie"].mean() if len(subset) > 0 else np.nan,
                "bleu_brie_sum": subset["bleu_brie_sum"].mean() if len(subset) > 0 else np.nan,
                "bleu_random": subset["bleu_random"].mean() if len(subset) > 0 else np.nan,
                "bleu_cnt": subset["bleu_cnt"].mean() if len(subset) > 0 else np.nan,

                # ROUGE
                "rouge_brie": subset["rouge_brie"].mean() if len(subset) > 0 else np.nan,
                "rouge_brie_sum": subset["rouge_brie_sum"].mean() if len(subset) > 0 else np.nan,
                "rouge_random": subset["rouge_random"].mean() if len(subset) > 0 else np.nan,
                "rouge_cnt": subset["rouge_cnt"].mean() if len(subset) > 0 else np.nan,

                # DIVERSITY
                "dist1_brie": subset["dist1_brie"].mean() if len(subset) > 0 else np.nan,
                "dist1_brie_sum": subset["dist1_brie_sum"].mean() if len(subset) > 0 else np.nan,
                "dist1_random": subset["dist1_random"].mean() if len(subset) > 0 else np.nan,
                "dist1_cnt": subset["dist1_cnt"].mean() if len(subset) > 0 else np.nan,

                "dist2_brie": subset["dist2_brie"].mean() if len(subset) > 0 else np.nan,
                "dist2_brie_sum": subset["dist2_brie_sum"].mean() if len(subset) > 0 else np.nan,
                "dist2_random": subset["dist2_random"].mean() if len(subset) > 0 else np.nan,
                "dist2_cnt": subset["dist2_cnt"].mean() if len(subset) > 0 else np.nan,

                # COVERAGE
                "cov_brie": subset["cov_brie"].mean() if len(subset) > 0 else np.nan,
                "cov_brie_sum": subset["cov_brie_sum"].mean() if len(subset) > 0 else np.nan,
                "cov_random": subset["cov_random"].mean() if len(subset) > 0 else np.nan,
                "cov_cnt": subset["cov_cnt"].mean() if len(subset) > 0 else np.nan,

                # LENGTH
                "len_brie": subset["len_brie"].mean() if len(subset) > 0 else np.nan,
                "len_brie_sum": subset["len_brie_sum"].mean() if len(subset) > 0 else np.nan,
                "len_random": subset["len_random"].mean() if len(subset) > 0 else np.nan,
                "len_cnt": subset["len_cnt"].mean() if len(subset) > 0 else np.nan,
            })

        df_metrics_by_photos = pd.DataFrame(rows)

        bleu_figure_data["metrics"].extend([
            {
                "model_name": "BRIE",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_bleu": df_metrics_by_photos["bleu_brie"].tolist(),
            },
            {
                "model_name": "BRIE+SUM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_bleu": df_metrics_by_photos["bleu_brie_sum"].tolist(),
            },
            {
                "model_name": "RANDOM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_bleu": df_metrics_by_photos["bleu_random"].tolist(),
            },
            {
                "model_name": "CNT",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_bleu": df_metrics_by_photos["bleu_cnt"].tolist(),
            }
        ])

        rouge_figure_data["metrics"].extend([
            {
                "model_name": "BRIE",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_rouge": df_metrics_by_photos["rouge_brie"].tolist(),
            },
            {
                "model_name": "BRIE+SUM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_rouge": df_metrics_by_photos["rouge_brie_sum"].tolist(),
            },
            {
                "model_name": "RANDOM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_rouge": df_metrics_by_photos["rouge_random"].tolist(),
            },
            {
                "model_name": "CNT",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_rouge": df_metrics_by_photos["rouge_cnt"].tolist(),
            }
        ])

        dist1_figure_data["metrics"].extend([
            {
                "model_name": "BRIE",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist1": df_metrics_by_photos["dist1_brie"].tolist(),
            },
            {
                "model_name": "BRIE+SUM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist1": df_metrics_by_photos["dist1_brie_sum"].tolist(),
            },
            {
                "model_name": "RANDOM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist1": df_metrics_by_photos["dist1_random"].tolist(),
            },
            {
                "model_name": "CNT",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist1": df_metrics_by_photos["dist1_cnt"].tolist(),
            }
        ])

        dist2_figure_data["metrics"].extend([
            {
                "model_name": "BRIE",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist2": df_metrics_by_photos["dist2_brie"].tolist(),
            },
            {
                "model_name": "BRIE+SUM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist2": df_metrics_by_photos["dist2_brie_sum"].tolist(),
            },
            {
                "model_name": "RANDOM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist2": df_metrics_by_photos["dist2_random"].tolist(),
            },
            {
                "model_name": "CNT",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_dist2": df_metrics_by_photos["dist2_cnt"].tolist(),
            }
        ])

        cov_figure_data["metrics"].extend([
            {
                "model_name": "BRIE",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_cov": df_metrics_by_photos["cov_brie"].tolist(),
            },
            {
                "model_name": "BRIE+SUM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_cov": df_metrics_by_photos["cov_brie_sum"].tolist(),
            },
            {
                "model_name": "RANDOM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_cov": df_metrics_by_photos["cov_random"].tolist(),
            },
            {
                "model_name": "CNT",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_cov": df_metrics_by_photos["cov_cnt"].tolist(),
            }
        ])

        len_figure_data["metrics"].extend([
            {
                "model_name": "BRIE",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_len": df_metrics_by_photos["len_brie"].tolist(),
            },
            {
                "model_name": "BRIE+SUM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_len": df_metrics_by_photos["len_brie_sum"].tolist(),
            },
            {
                "model_name": "RANDOM",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_len": df_metrics_by_photos["len_random"].tolist(),
            },
            {
                "model_name": "CNT",
                "min_photos": df_metrics_by_photos["min_photos"].tolist(),
                "mean_len": df_metrics_by_photos["len_cnt"].tolist(),
            }
        ])

        output_metrics_photos = f"docs/{datamodule.city}/metrics_by_photos_{model}.csv"
        df_metrics_by_photos.to_csv(output_metrics_photos, index=False)

        print(f"Metrics by photos saved to: {output_metrics_photos}")

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

        #percentile_figure_data["metrics"].append(model_percentile_metrics)

       # For the recall metric, only include users with >= train images

        test_cases = test_cases[test_cases["author_num_train_photos"] >= 10]

        # Initialize recall table data
        model_recall_metrics = {"k": [], "Recall@10": [], "model_name": model}
        model_ndcg_metrics = {"k": [], "NDCG@10": [], "model_name": model}

        filtered_test_set = test_set[
            test_set["id_test"].isin(test_cases["id_test"])
        ].reset_index(drop=True)

        if filtered_test_set.empty:
            print("No test cases after filtering → skipping Recall/NDCG/BLEU")
            continue

        print("")

        preds = torch.tensor(filtered_test_set["pred"], dtype=torch.float)
        target = torch.tensor(filtered_test_set["is_dev"], dtype=torch.long)
        indexes = torch.tensor(filtered_test_set["id_test"], dtype=torch.long)
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

        #recall_figure_data["metrics"].append(model_recall_metrics)
        #ndcg_figure_data["metrics"].append(model_ndcg_metrics)

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
    #figures.percentile_figure(percentile_figure_data)
    figures.bleu_figure(bleu_figure_data)
    figures.rouge_figure(rouge_figure_data)
    figures.generic_metric_figure(dist1_figure_data, "mean_dist1", "Distinct-1", "dist1")
    figures. generic_metric_figure(dist2_figure_data, "mean_dist2", "Distinct-2", "dist2")
    figures.generic_metric_figure(cov_figure_data, "mean_cov", "Coverage", "coverage")
    figures.generic_metric_figure(len_figure_data, "mean_len", "Length Ratio", "length")
