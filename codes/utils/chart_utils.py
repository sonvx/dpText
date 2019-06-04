import matplotlib.pyplot as plt


def draw_line_charts_fig2left_wordlevel(nodp_vs_gold_100, dp_vs_gold_100):

    # title = "Semantic space changes between DP and NoDP using MAP metrics."
    x_axis_labels = ["20", '200', '500', '1K', '5K', '10K', '50K', '90K', '100K']
    x_axis_position = [i for i in range(len(x_axis_labels))]

    print(x_axis_position)

    fig, ax = plt.subplots()
    ax.plot(x_axis_position, nodp_vs_gold_100, 'o-', color="g", label="None-DP Embedding")
    ax.plot(x_axis_position, dp_vs_gold_100, '+-', color="r", label="DP Embedding")
    ax.set(xlabel='Learning Step', ylabel='MAP Score')
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.xticks(x_axis_position, x_axis_labels, rotation=70)  # 'vertical')
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.05)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.legend(loc="best")
    # fig.savefig("nodp_vs_DP_topKwords.png")
    fig.savefig("fig2_left_nodp_and_DP_to_golds_topKwords_wordlevel.pdf", bbox_inches='tight')


def draw_line_charts_fig2right_charlevel(nodp_vs_gold_100, dp_vs_gold_100):
    # title = "Semantic space changes between DP and NoDP using MAP metrics."
    x_axis_labels = ["20", '200', '500', '1K', '5K', '10K', '50K', '90K', '100K']
    x_axis_position = [i for i in range(len(x_axis_labels))]

    print(x_axis_position)

    fig, ax = plt.subplots()
    ax.plot(x_axis_position, nodp_vs_gold_100, 'o-', color="g", label="None-DP Embedding")
    ax.plot(x_axis_position, dp_vs_gold_100, '+-', color="r", label="DP Embedding")
    ax.set(xlabel='Learning Step', ylabel='MAP Score')
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.xticks(x_axis_position, x_axis_labels, rotation=70)  # 'vertical')
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.05)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.legend(loc="best")
    fig.savefig("fig2_right_nodp_and_DP_to_golds_topKwords_charectorlevel.pdf", bbox_inches='tight')
