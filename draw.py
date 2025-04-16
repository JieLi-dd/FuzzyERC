import json
import matplotlib.pyplot as plt
import numpy as np



def plot_softmax_comparison(softmax_text, softmax_audio, softmax_video, index, class_names=None, case=0):
    """
    绘制指定样本在三个模态下的 softmax 概率分布柱状图。

    参数:
        softmax_text: list of list, 文本模态 softmax 输出
        softmax_audio: list of list, 音频模态 softmax 输出
        softmax_video: list of list, 视频模态 softmax 输出
        index: int, 指定样本的下标
        class_names: list of str, 类别名（可选），长度应与类别数一致
    """
    prob_text = np.array(softmax_text[index])
    prob_audio = np.array(softmax_audio[index])
    prob_video = np.array(softmax_video[index])

    num_classes = len(prob_text)
    x = np.arange(num_classes)

    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width, prob_text, width=bar_width, label='Text', color='skyblue')
    plt.bar(x, prob_video, width=bar_width, label='Video', color='lightgreen')
    plt.bar(x + bar_width, prob_audio, width=bar_width, label='Audio', color='salmon')

    plt.xlabel('Class Index' if class_names is None else 'Class')
    plt.ylabel('Probability')
    plt.title(f'Softmax Probability Comparison for Sample Index {index}')
    plt.xticks(x, class_names if class_names else x)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'./results/{case}/softmax_comparison_{index}.png')


def collect_abnormal_indices(case_indices, labels, preds_text, preds_video, preds_audio,
                             softmax_text, softmax_video, softmax_audio):
    """
    在每个 case 中，找出“预测正确模态在真实类别上的概率反而小于错误模态预测类别的概率”的样本下标。

    返回：
        case_abnormal_indices: List[List[int]], 每个 case 中异常样本的下标列表
    """
    case_abnormal_indices = [[] for _ in range(8)]

    for case_id, indices in enumerate(case_indices):
        if case_id == 0 or case_id == 7:
            continue  # 跳过全对或全错

        for i in indices:
            label = labels[i]
            pt, pv, pa = preds_text[i], preds_video[i], preds_audio[i]
            st, sv, sa = softmax_text[i], softmax_video[i], softmax_audio[i]

            # 获取正确模态在真实标签上的 softmax 值
            correct_scores = []
            if case_id in [1, 2, 3]:  # text 正确
                correct_scores.append(st[label])
            if case_id in [1, 4, 5]:  # video 正确
                correct_scores.append(sv[label])
            if case_id in [2, 4, 6]:  # audio 正确
                correct_scores.append(sa[label])

            max_correct_score = max(correct_scores)

            # 检查错误模态在其预测标签上的 softmax 是否 > 正确模态真实标签上的 softmax
            abnormal = False
            if case_id in [4, 5, 6]:  # text 错误
                if st[pt] > max_correct_score:
                    abnormal = True
            if case_id in [2, 3, 6]:  # video 错误
                if sv[pv] > max_correct_score:
                    abnormal = True
            if case_id in [1, 3, 5]:  # audio 错误
                if sa[pa] > max_correct_score:
                    abnormal = True

            if abnormal:
                case_abnormal_indices[case_id].append(i)

    return case_abnormal_indices

def count_max_prob_modal_per_case(case_indices, softmax_text, softmax_video, softmax_audio):
    """
    统计每个 case 中，三个模态中哪个模态 softmax 概率最大。

    参数：
        case_indices: List[List[int]], 每个 case 中的样本下标列表
        softmax_text, softmax_video, softmax_audio: List[List[float]], 每个模态的 softmax 概率

    返回：
        case_modal_max_counts: List[Dict[str, int]], 每个 case 中 text/audio/video 最大数量
    """
    case_modal_max_counts = []

    for indices in case_indices:
        count_text, count_audio, count_video = 0, 0, 0

        for i in indices:
            st_max = max(softmax_text[i])
            sa_max = max(softmax_audio[i])
            sv_max = max(softmax_video[i])

            if st_max >= sa_max and st_max >= sv_max:
                count_text += 1
            elif sa_max >= st_max and sa_max >= sv_max:
                count_audio += 1
            else:
                count_video += 1

        case_modal_max_counts.append({
            "text_max_count": count_text,
            "audio_max_count": count_audio,
            "video_max_count": count_video
        })

    return case_modal_max_counts


if __name__ == '__main__':
    path_text = './results/softmax_list_text.json'
    path_video = './results/softmax_list_video.json'
    path_audio = './results/softmax_list_audio.json'

    with open(path_text, 'r') as f:
        text_results = json.load(f)
    labels = text_results['labels']
    preds_text = text_results['preds']
    softmax_text = text_results['softmax']

    with open(path_video, 'r') as f:
        video_results = json.load(f)
    preds_video = video_results['preds']
    softmax_video = video_results['softmax']

    with open(path_audio, 'r') as f:
        audio_results = json.load(f)
    preds_audio = audio_results['preds']
    softmax_audio = audio_results['softmax']

    case_indices = [[] for _ in range(8)]  # 每种情况一个列表

    for i in range(len(labels)):
        correct_t = preds_text[i] == labels[i]
        correct_v = preds_video[i] == labels[i]
        correct_a = preds_audio[i] == labels[i]

        # 构建三位二进制对应情况索引，例如 111 -> 7_0, 110 -> 6_1, 101 -> 5_2, 100 -> 4_3, 011 -> 3_4, 010 -> 2_5, 001 -> 1_6, 000 -> 0_7
        bits = (correct_t << 2) | (correct_v << 1) | (correct_a)
        case_idx = 7 - bits  # 将 111 对应 0，000 对应 7
        case_indices[case_idx].append(i)

    # for idx, indices in enumerate(case_indices):
    #     print(f"Case {idx}: {len(indices)} samples")
    #     print(indices)
    #     if idx < 7:
    #         continue
    #     for i in indices:
    #         plot_softmax_comparison(
    #             softmax_text=softmax_text,
    #             softmax_audio=softmax_audio,
    #             softmax_video=softmax_video,
    #             index=i,
    #             class_names=["Neutral", "Frustrated", "Angry", "Sad", "Happy", "Excited"],
    #             case=idx
    #         )

    abnormal_indices_per_case = collect_abnormal_indices(
        case_indices=case_indices,
        labels=labels,
        preds_text=preds_text,
        preds_video=preds_video,
        preds_audio=preds_audio,
        softmax_text=softmax_text,
        softmax_video=softmax_video,
        softmax_audio=softmax_audio
    )

    # 打印每个 case 中异常样本的下标数量
    for i, indices in enumerate(abnormal_indices_per_case):
        print(f"Case {i}: {len(indices)} abnormal samples, indices: {indices[:5]}{'...' if len(indices) > 5 else ''}")

    case_modal_max_counts = count_max_prob_modal_per_case(case_indices=case_indices, softmax_text=softmax_text, softmax_video=softmax_video, softmax_audio=softmax_audio)
    print(case_modal_max_counts)


    print('done')