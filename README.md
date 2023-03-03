## Universal Style Transfer

This is the Pytorch Diversified-Arbitrary-Style-Transfer-via-Deep-Feature-Perturb.

based on https://github.com/ChauvinisticJonatha/Implementation-of-the-paper-Diversified-Arbitrary-Style-Transfer-via-Deep-Feature-Perturb.git

and use pytorch==1.12 instead of 0.4


## Prerequisites
```
pip install -r requirements.txt
```

## Prepare images
Put content and image pairs in `images/content` and `images/style` respectively. Note that correspoding conternt and image pairs should have same names.


## Style Transfer

```
python WCT.py --cuda
```



### Reference
Li Y, Fang C, Yang J, et al. Universal Style Transfer via Feature Transforms[J]. arXiv preprint arXiv:1705.08086, 2017.

Wang Z, Zhao L, Chen H, et al. Diversified arbitrary style transfer via deep feature perturbation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 7789-7798.
