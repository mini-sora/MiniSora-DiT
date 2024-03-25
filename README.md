# Mini Sora 社区 MiniSora-DiT 复现项目

MiniSora-DiT, a DiT reproduction based on XTuner from the open source community MiniSora

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Stargazers][stars-shield]][stars-url]
<br />

<!-- PROJECT LOGO -->
<div align="center">

<img src="assets/logo.jpg" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
  </div>
</div>

<div align="center">

[English](README_EN.md) | 简体中文  

</div>

<p align="center">
    👋 加入我们的 <a href="https://cdn.vansin.top/minisora.jpg" target="_blank">微信社区</a>
</p>

Mini Sora 开源社区定位为由社区同学自发组织的开源社区（**免费不收取任何费用、不割韭菜**），Mini Sora 计划探索 Sora 的实现路径和后续的发展方向：

- 将定期举办 Sora 的圆桌和社区一起探讨可能性
- 视频生成的现有技术路径探讨

## MiniSora社区复现小组

[**MiniSora复现小组页面**](https://github.com/mini-sora/minisora/tree/main/codes)

### MiniSora-DiT: 基于XTuner复现论文DiT

#### 招募要求

招募MiniSora社区同学使用 `XTuner` 复现 `DiT`, 希望领取任务同学有如下特点：

1. 熟悉 `OpenMMLab MMEngine` 机制
2. 熟悉 `DiT`

#### 背景

1. `DiT` 作者和 `Sora` 作者为同一个
2. `XTuner` 现有能够高效训练 `1000K` 序列长度的核心技术

#### 支持

1. 算力提供 2*A100
2. XTuner 核心开发者 [P佬@pppppM](https://github.com/pppppM) 会大力支持~

[**XTuner**: https://github.com/internLM/xtuner](https://github.com/internLM/xtuner)

## 最近更新

- [**empty**:](empty)


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

目前已在 dev 分支提交了 DiT 在纯 torch 下的复现代码 [fast-DiT](https://github.com/chuanyangjin/fast-DiT)，该版本使用了混合精度还有一些加速方案，可以极大程度降低显存，以及提升训练速度。

1. 环境安装

使用 dev 分支中的 `environment.yml` 可以复现环境

```bash
conda env create -f environment.yml
conda activate DiT
```

2. 数据集预处理

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

3. 使用 mmengine 重写数据流，下面是原版的 dataset，可见直接读取上一步生成的 npy 文件，省去了前处理时间

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

4. 重写 loss 计算
5. 使用 xtuner 调训练 pipeline

## 论文共读计划

### 论文共读发表者募集

- [**DiT** (ICCV 23 Paper)](https://github.com/orgs/mini-sora/discussions/39)
- [**Stable Cascade** (ICLR 24 Paper)](https://github.com/orgs/mini-sora/discussions/145)

## Sora复现小组-MiniSora社区微信交流群

<div align="center">

<img src="./assets/sora-reproduce.png" width="200"/>
  <div>&nbsp;</div>
  <div align="center">
  </div>
</div>

## Mini Sora 微信社区社区交流群

<div align="center">
<img src="assets/qrcode.png" width="200"/>

  <div>&nbsp;</div>
  <div align="center">
  </div>
</div>

## MiniSora Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mini-sora/minisora&type=Date)](https://star-history.com/#mini-sora/minisora&Date)

## 如何向Mini Sora 社区贡献

我们非常希望你们能够为 Mini Sora 开源社区做出贡献，并且帮助我们把它做得比现在更好！

具体查看[贡献指南](./docs/CONTRIBUTING.md)

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

