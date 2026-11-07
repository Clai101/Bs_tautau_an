"""Построить форму сигнала после отбора BDT."""

from bdt_header import (
    argparse, json, Path, np, ds, plt, XGBClassifier, hist_approx, json_update,
    mask_physics, mask_mc_signal, mask_mc_tag,
    channel_selections, model_groups, channel_labels, cut_aliases,
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
    args = parser.parse_args(argv)
    channel = args.channel
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
    selection = mask_physics & channel_selections[channel] & mask_mc_signal & mask_mc_tag
    files = sorted((root / "new_sig").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("Нет входных Parquet-файлов")
    dataset = ds.dataset(files, format="parquet")
    read_columns = list(dict.fromkeys(columns + ["E_gamma_in_ROE", "MBtag"]))
    data = dataset.to_table(columns=read_columns, filter=selection).to_pandas()
    if data.empty:
        raise ValueError(f"После базового отбора нет событий: {channel}")
    scores = model.predict_proba(data[columns])[:, 1]
    selected = data.loc[scores > cut]

    # 4. Сохраняем исходную схему биннинга и формат hist_approx.
    width_x, width_y = 0.06, 0.0075
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

    output = args.output_dir.resolve()
    for directory in ("shapes", "quantity", "output"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    histogram.save_hist(str(output / "shapes" / f"sig_{channel}.json"))
    json_update(str(output / "quantity" / f"N_{channel}.json"), {"N_sig": len(selected)})

    # 5. Один итоговый график.
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(bins_x, bins_y, counts.T, shading="auto", cmap=PLOT_CMAPS["signal"])
    ax.set_xlabel(r"$E_\gamma^{\mathrm{ROE}}\ \mathrm{GeV}$")
    ax.set_ylabel(r"$M_{B_s^0}\ \mathrm{GeV}$")
    ax.set_title(rf"Signal $B_s^0 \to \tau\tau \to {channel_labels[channel]}$")
    fig.colorbar(mesh, ax=ax, label="Counts")
    fig.tight_layout()
    fig.savefig(output / "output" / f"sig_E_ROE_M_Bs{channel}.pdf", dpi=700)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"sig/{channel}: cut={cut:.6f}, before={len(data)}, "
          f"after={len(selected)}, in histogram={int(counts.sum())}")


if __name__ == "__main__":
    main()
