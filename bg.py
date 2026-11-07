"""Построить форму фона по четвёртой части generic MC."""

from bdt_header import (
    argparse, json, Path, np, pd, ds, plt, XGBClassifier, hist_approx, json_update,
    mask_physics,
    channel_selections, model_groups, channel_labels, cut_aliases,
    classify_background_source, BACKGROUND_FIT_GROUPS, BACKGROUND_LABELS,
    PLOT_CMAPS,
)


def main(argv=None):
    # 1. Параметры запуска; входные данные отсчитываются от каталога скрипта.
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel", choices=tuple(channel_selections))
    parser.add_argument("--output-dir", type=Path, default=root,
                        help="Каталог с подкаталогами shapes, quantity и output")
    parser.add_argument("--show", action="store_true", help="Показать график после сохранения")
    parser.add_argument("--no-latex", action="store_true", help="Рисовать без внешнего LaTeX")
    parser.add_argument("--background-scale", type=float, default=4,
                        help="Множитель N_bg для экстраполяции части 4 (по умолчанию 4)")
    args = parser.parse_args(argv)
    channel = args.channel
    if not np.isfinite(args.background_scale) or args.background_scale <= 0:
        parser.error("--background-scale должен быть конечным положительным числом")
    if not args.show:
        plt.switch_backend("Agg")
    if args.no_latex:
        plt.rcParams.update({"text.usetex": False, "font.serif": ["DejaVu Serif"]})

    # 2. Модель и порог: метаданные читаются один раз.
    model_group = model_groups[channel]
    with (root / "models" / f"columns_and_fom_{model_group}.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    columns = metadata["columns"]
    cuts = metadata["FoM"]
    cut_key = channel if channel in cuts else cut_aliases.get(channel, channel)
    cut = float(cuts[cut_key])
    if not np.isfinite(cut) or not 0 <= cut <= 1:
        raise ValueError(f"Некорректный порог BDT для {channel}: {cut}")
    model = XGBClassifier()
    model.load_model(root / "models" / f"bdt_model_{model_group}.json")

    # 3. Маска собирается из блоков bdt_header.
    selection = mask_physics & channel_selections[channel]
    files = sorted((root / "gen_data").glob("*4.parquet"))
    if not files:
        raise FileNotFoundError("Нет входных Parquet-файлов")
    read_columns = list(dict.fromkeys(columns + ["E_gamma_in_ROE", "MBtag", "is0"]))
    parts = []
    for path in files:
        part = ds.dataset(path, format="parquet").to_table(
            columns=read_columns, filter=selection,
        ).to_pandas()
        part["background_source"] = classify_background_source(path, part["is0"])
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    if data.empty:
        raise ValueError(f"После базового отбора нет событий: {channel}")
    scores = model.predict_proba(data[columns])[:, 1]
    selected = data.loc[scores > cut]

    # Две фоновые формы старого toy_mc: правильно восстановленный тег в BsBs
    # и объединение всех остальных источников/тегов. Не смешиваем τ-каналы.
    is_bsbs = selected["background_source"].isin(BACKGROUND_FIT_GROUPS["bsbs"])
    components = {"bsbs": selected.loc[is_bsbs], "ubs": selected.loc[~is_bsbs]}

    # 4. Сохраняем исходную схему биннинга и формат hist_approx.
    width_x, width_y = {"pilep": (0.4, 0.05), "rholep": (0.2, 0.05)}.get(channel, (0.15, 0.025))
    bins_x = np.linspace(0, 1.2, int(1.2 // width_x + 2))
    bins_y = np.linspace(5.25, 5.5, int((5.5 - 5.25) // width_y + 2))
    counts, _ = np.histogramdd(selected[["E_gamma_in_ROE", "MBtag"]].to_numpy(),
                               bins=(bins_x, bins_y))
    if counts.sum() == 0:
        raise ValueError(f"Пустая гистограмма после BDT: {channel}, cut={cut}")
    centers = np.meshgrid((bins_x[:-1] + bins_x[1:]) / 2,
                          (bins_y[:-1] + bins_y[1:]) / 2, indexing="ij")
    histogram = hist_approx()
    histogram.put_hist(centers, counts)

    component_histograms = {}
    component_counts = {}
    for name, subset in components.items():
        component_counts[name], _ = np.histogramdd(
            subset[["E_gamma_in_ROE", "MBtag"]].to_numpy(), bins=(bins_x, bins_y),
        )
        if component_counts[name].sum() == 0:
            raise ValueError(f"Пустая компонента {name}: {channel}; PDF нельзя нормировать")
        component_histograms[name] = hist_approx()
        component_histograms[name].put_hist(centers, component_counts[name])
    np.testing.assert_array_equal(counts, component_counts["bsbs"] + component_counts["ubs"])

    output = args.output_dir.resolve()
    for directory in ("shapes", "quantity", "output"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    histogram.save_hist(str(output / "shapes" / f"bg_{channel}.json"))
    for name, component_histogram in component_histograms.items():
        component_histogram.save_hist(str(output / "shapes" / f"bg_{name}_{channel}.json"))
        if channel == "lep":
            # Имена файлов, которые читает old/toy_mc.ipynb.
            component_histogram.save_hist(str(output / "output" / f"{name}.json"))
    json_update(str(output / "quantity" / f"N_{channel}.json"), {
        "N_bg": len(selected) * args.background_scale,
        "N_bg_bsbs": len(components["bsbs"]) * args.background_scale,
        "N_bg_ubs": len(components["ubs"]) * args.background_scale,
        "background_scale": args.background_scale,
        "background_components": {
            "bsbs": "bsbs MC with is0 == 1",
            "ubs": "all remaining events, including undefined is0",
        },
    })

    # 5. Общий фон и обе компоненты: заголовок и цвет соответствуют содержимому.
    for name, plot_counts in {"background": counts, **component_counts}.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(bins_x, bins_y, plot_counts.T, shading="auto",
                             cmap=PLOT_CMAPS[name])
        ax.set_xlabel(r"$E_\gamma^{\mathrm{ROE}}\ \mathrm{GeV}$")
        ax.set_ylabel(r"$M_{B_s^0}\ \mathrm{GeV}$")
        label = "Total background" if name == "background" else f"Background: {BACKGROUND_LABELS[name]}"
        ax.set_title(rf"{label}, $\tau\tau\to {channel_labels[channel]}$")
        fig.colorbar(mesh, ax=ax, label="Counts")
        fig.tight_layout()
        filename = (f"bg_E_ROE_M_Bs_{channel}.pdf" if name == "background"
                    else f"bg_{name}_E_ROE_M_Bs_{channel}.pdf")
        fig.savefig(output / "output" / filename, dpi=700)
        if args.show:
            plt.show()
        plt.close(fig)
    print(f"bg/{channel}: cut={cut:.6f}, before={len(data)}, "
          f"after={len(selected)}, in histogram={int(counts.sum())}, "
          f"bsbs={len(components['bsbs'])}, ubs={len(components['ubs'])}")


if __name__ == "__main__":
    main()
