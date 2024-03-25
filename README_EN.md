# Mini Sora 社区Sora复现小组

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Stargazers][stars-shield]][stars-url]
<br />

<!-- PROJECT LOGO -->
<div align="center">

<img src="./assets/logo.jpg" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
  </div>
</div>

<div align="center">

English | [简体中文](./README_CN.md)  

</div>

## Mini Sora 复现目标

1. **GPU-Friendly**: 最好对GPU内存大小和GPU数量要求较低,比如8卡A100,4KA6000,单卡Rtx4090之类的算力可以训练和推理
2. **Training-Efficiency** : 不需要训练太久即可有较好的效果
3. **Inference-Efficiency**: 推理生成视频时, 长度和分辨率不要求过高, 如3-10s,480p都是可接受的

候选复现论文主要有以下三篇, 来作为后续Sora复现的Baseline, 社区已经(02/29)将[OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT)和[SiT](https://github.com/willisma/SiT)代码Fork到codes文件夹下, 期待贡献者提交PR, 将Baseline代码迁移到Sora复现工作上来. [**Update**] 03/02, 添加[StableCascade](https://github.com/Stability-AI/StableCascade) codes

- DiT with **OpenDiT**
  - OpenDiT采用分布式训练，生成图片采用8卡A100训练
  - 另外社区小伙伴咨询了OpenDiT负责人，得到如下信息
    - 采用百卡，训练一个月到3月15号可以出更多的测试视频
    - 用OpenDiT跑Sora，他估计最少要千卡，如果只有很少的卡也能训练起来，但是效果非常差，如果卡不够，一个视频做token时，只能丢弃一些，在扩散生成的时候，视频还原度很低
  - OpenDiT采用的sd的vae编码，采用的是sd的预训练模型，可能根本不适合视频生成数据集，Sora应该是重新训练了vae，压缩率高了
  - Sora Leader做过DALLE3，生成视频 的 解码 是用类似DALLE3的扩散方式, 所以压缩编码的时候应该是DALLE3的 反方向的方式
- **SiT**
- **W.A.L.T**(还未release)
- **StableCascade**
  - ToDo: make it as a video-based model with additional temp layer in the near future

## 数据集

- ImageNet-1K

可以在 OpenDataLab 进行下载 [ImageNet-1K](https://opendatalab.org.cn/OpenDataLab/ImageNet-1K)

```shell
pip install openxlab #安装
pip install -U openxlab #版本升级
openxlab login #进行登录，输入对应的AK/SK

cd ${dataset_dir}
openxlab dataset get --dataset-repo OpenDataLab/ImageNet-1K #数据集下载
```

## 复现步骤

1. 数据集预处理

因为在原版 Meta 的 [DiT](https://github.com/facebookresearch/DiT) 中，每个 iter 都会对数据进行重复计算，为了节省训练的时间，可以先对图片进行预处理，在训练的时候可以节省这部分的时间

详见 dev 分支中的 [extract_features.py#L163](https://github.com/mini-sora/MiniSora-DiT/blob/ad13c58370842db333c77253709e3fbbc1e9a092/extract_features.py#L163-L177) ，处理需要时间较久，大概 1~2小时。

```python
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
        x = x.detach().cpu().numpy()    # (1, 4, 32, 32)
        np.save(f'{args.features_path}/imagenet256_features/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.features_path}/imagenet256_labels/{train_steps}.npy', y)
            
        train_steps += 1
        print(train_steps)
```

执行后会对每个图片生成一个 npy 文件，训练的时候直接读取

2. 使用 mmengine 重写数据流，下面是原版的 dataset，可见直接读取上一步生成的 npy 文件，省去了前处理时间

```python
class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)
```

3. 重写 loss 计算
4. 使用 xtuner 调训练 pipeline

## 模型架构

...

## 算力需求

论文原版是：`8 x A100` ，但是使用混合精度，可以使用 `2 x A100`

## 其他项目

- 非Sora一键文本生成视频 : [Text2Video](./Others/Text2Video.md)
  - 项目介绍: 这是一个文本转视频的项目，通过输入文本，一键直接生成对应的视频。
  - 技术栈：
    - 文本处理，分割文本，生成 prompt
    - 语音合成，将文本转换为语音 text to speech (TTS)，azure speech
    - 图片生成，将文本转成图片，openai DALL·E
    - 视频合成，将图片和语音合成视频，moviepy
  - 项目要求:
    - openai key，用 DALL·E 生成图片；
    - azure speech key，将文本转成语音。
  
<!-- 
**提交PR或者Issue后**, 可以申请加入MiniSora贡献者社群并申请加入 Sora 有关论文复现小组！

<div align="center">

<img src="assets/sora-reproduce.png" width="200"/>
  <div>&nbsp;</div>
  <div align="center">
  </div>
</div>
-->

## Sora复现小组-MiniSora社区微信交流群

<div align="center">

<img src="../assets/sora-reproduce.png" width="200"/>
  <div>&nbsp;</div>
  <div align="center">
  </div>
</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mini-sora/minisora&type=Date)](https://star-history.com/#mini-sora/minisora&Date)

## 如何向Mini Sora 社区贡献

我们非常希望你们能够为 Mini Sora 开源社区做出贡献，并且帮助我们把它做得比现在更好！

具体查看[贡献指南](../docs/CONTRIBUTING.md)

## 社区贡献者

<!-- readme: collaborators,contributors -start -->

<!-- readme: collaborators,contributors -end -->

<a href="https://github.com/mini-sora/minisora/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mini-sora/minisora" />
</a>

[your-project-path]: mini-sora/minisora
[contributors-shield]: https://img.shields.io/github/contributors/mini-sora/minisora.svg?style=flat-square
[contributors-url]: https://github.com/mini-sora/minisora/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/mini-sora/minisora.svg?style=flat-square
[forks-url]: https://github.com/mini-sora/minisora/network/members
[stars-shield]: https://img.shields.io/github/stars/mini-sora/minisora.svg?style=flat-square
[stars-url]: https://github.com/mini-sora/minisora/stargazers
[issues-shield]: https://img.shields.io/github/issues/mini-sora/minisora.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/mini-sora/minisora.svg
[license-shield]: https://img.shields.io/github/license/mini-sora/minisora.svg?style=flat-square
[license-url]: https://github.com/mini-sora/minisora/blob/main/LICENSE