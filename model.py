import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.norm = nn.LayerNorm(hidden_size)  # 可选
        self.beta = nn.Parameter(torch.ones(1, hidden_size))  # 可学习权重
        if dataset == 'MELD':
            self.fc[0].weight.data.copy_(torch.eye(hidden_size))
            self.fc[0].weight.requires_grad = False
            self.fc[2].weight.data.copy_(torch.eye(hidden_size))
            self.fc[2].weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.norm(self.fc(a)))  # 归一化门控
        final_rep = (self.beta * z) * a  # 学习缩放权重
        return final_rep


class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        """
        :param a: (batch_size, len, dim)
        :param b: (batch_size, len, dim)
        :param c: (batch_size, len, dim)
        :return: (batch_size, len, dim)
        """
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        # 计算最大值的索引 (batch_size, len, dim)，索引是 modal_size 维度上的最大值索引
        max_indices = torch.argmax(utters_fc, dim=-2, keepdim=True)  # (batch_size, len, 1, dim)

        # 使用 gather 选择最大值对应的模态
        final_rep = torch.gather(utters, dim=-2, index=max_indices).squeeze(-2)  # (batch_size, len, dim)

        return final_rep


class Unimodal_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_input, n_head, n_classes, hidden_dim, n_speakers, dropout):
        super(Unimodal_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx)

        # Temporal convolutional layers
        self.modal_input = nn.Conv1d(D_input, hidden_dim, kernel_size=1, padding=0, bias=False)

        # Self-Transformers
        self.modal_intra = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.modal_intra_fusion = Unimodal_GatedFusion(hidden_size=hidden_dim, dataset=dataset)

    def forward(self, x_input,  u_mask, qmask, dia_len):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        spk_embeddings = self.speaker_embeddings(spk_idx)

        # Temporal convolutional layers
        x_input = self.modal_input(x_input.permute(1, 2, 0)).transpose(1, 2)
        x_input = self.modal_intra(x_input, x_input, u_mask, spk_embeddings)
        x_input = self.modal_intra_fusion(x_input)

        return x_input


class Multimodal_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
        super(Multimodal_Based_Model, self).__init__()
        self.text = Unimodal_Based_Model(dataset, temp, D_text, n_head, n_classes, hidden_dim, n_speakers, dropout)
        self.audio = Unimodal_Based_Model(dataset, temp, D_audio, n_head, n_classes, hidden_dim, n_speakers, dropout)
        self.video = Unimodal_Based_Model(dataset, temp, D_visual, n_head, n_classes, hidden_dim, n_speakers, dropout)

        self.fusion = Multimodal_GatedFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)



    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        text_logit, text_features = self.text(textf, u_mask, qmask, dia_len)
        audio_logit, audio_features = self.audio(acouf, u_mask, qmask, dia_len)
        video_logit, video_features = self.video(visuf, u_mask, qmask, dia_len)

        final_featrures = self.fusion( text_features, audio_features, video_features)
        final_logits = self.classifier(final_featrures)

        return final_logits, text_logit, audio_logit, video_logit, final_featrures, text_features, audio_features, video_features


class ConfidenceEstimator(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(ConfidenceEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 ∈ [0, 1]
        )
    def forward(self, h, logit):
        h = torch.cat([h, logit], dim=-1)
        return self.net(h)


class ConfidenceFusionModel(nn.Module):
    def  __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head, n_classes, hidden_dim, n_speakers, dropout):
        super(ConfidenceFusionModel, self).__init__()
        self.num_modalities = 3
        self.encoders = nn.ModuleList([
            Unimodal_Based_Model(dataset, temp, D, n_head, n_classes, hidden_dim, n_speakers, dropout)
            for D in [D_text, D_audio, D_visual]
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for _ in [D_text, D_audio, D_visual]
        ])
        self.conf_estimators = nn.ModuleList([
            ConfidenceEstimator(hidden_dim, n_classes) for _ in [D_text, D_audio, D_visual]
        ])
        self.fusion = nn.Linear(hidden_dim, n_classes)


    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, labels):
        input = [textf, acouf, visuf]
        hs, logits, weights, conf_losses = [], [], [], []
        hfs = []

        umask_bool = u_mask.bool()
        labels_ = labels[umask_bool]

        for x, enc, clf, conf_net in zip(input, self.encoders, self.classifiers, self.conf_estimators):
            h = enc(x, u_mask, qmask, dia_len)
            logit = clf(h)

            h = h[umask_bool]
            logit = logit[umask_bool]

            hs.append(h)
            logits.append(logit)

            #  计算置信度
            conf_score = conf_net(h, logit).squeeze(-1)
            weights.append(conf_score)

            if labels_ is not None:
                pred = logit.argmax(dim=-1)
                correct = (pred == labels_).float()
                conf_losses.append(F.binary_cross_entropy(conf_score, correct))

        # 权重归一化
        weights_tensor = torch.stack(weights, dim=1)  # [B, M]
        norm_weights = weights_tensor / (weights_tensor.sum(dim=1, keepdim=True) + 1e-6)

        # 加权融合
        hs_tensor = torch.stack(hs, dim=1)  # [B, M, D]
        fused = (norm_weights.unsqueeze(-1) * hs_tensor).sum(dim=1)  # [B, D]
        hfs.append(fused)

        # 最终融合预测
        out = self.fusion(fused)  # [B, C]

        ranking_loss = torch.tensor(0.0, device=out.device)
        if labels_ is not None:
            B, M = weights_tensor.shape
            corrects = torch.stack([
                (logit.argmax(dim=1) == labels_).float() for logit in logits
            ], dim=1)

            s_i = weights_tensor.unsqueeze(2)
            s_j = weights_tensor.unsqueeze(1)
            margin = s_i - s_j

            correct_i = corrects.unsqueeze(2)
            wrong_j = 1.0 - corrects.unsqueeze(1)
            valid_pair_mask = correct_i * wrong_j
            pos_margin = margin[valid_pair_mask.bool()]
            if pos_margin.numel() > 0:
                ranking_loss = F.relu(1.0 - pos_margin).mean()

        return hs, hfs, out, logits, weights_tensor, conf_losses, ranking_loss
