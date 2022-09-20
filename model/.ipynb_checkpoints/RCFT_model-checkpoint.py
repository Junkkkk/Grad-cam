import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import nn
import math
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

import sys
sys.path.append('/tf/workspace/Prediction/DL/models/EfficientNet_PyTorch')
sys.path.append('/tf/workspace/Prediction/DL/models/vit-pytorch')
sys.path.append('/tf/workspace/Prediction/DL/models/vision-master')
sys.path.append('/tf/workspace/Prediction/DL/models')

#from efficientnet_pytorch.model import EfficientNet
#from vit_pytorch import vit, cait, cross_vit
from TorchVision_color.models import resnet, densenet, inception


class Model(pl.LightningModule):
    def __init__(self, config, model_path):
        super().__init__()
        self.model_name = config.model_name
        self.l2_norm = config.l2_norm
        self.l1_norm = config.l1_norm
        self.model_path = model_path


        if self.model_name=='efficient-b0':
            hidden_dim = 128
            num_classes = 1
            dropout = 0.1
            self.net = EfficientNet.from_name(model_name='efficientnet-b0',)
                                                #override_params={'num_classes': 2},)
                                                #in_channels=1)
            self.classifier = nn.Sequential(
                nn.Linear(1280, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.model_name == 'resnet18':
            self.net = resnet.resnet18(pretrained=True)
            dim = 1000
            hidden_dim = 128
            num_classes = 2
            dropout = 0.1
            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )


        elif self.model_name=='resnet50':
            self.net = resnet.resnet50(pretrained=True)
            dim=1000
            hidden_dim = 128
            num_classes = 2
            dropout = 0.1
            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.model_name=='resnet101':
            self.net = resnet.resnet101(pretrained=True)
            dim=1000
            hidden_dim = 128
            num_classes = 2
            dropout = 0.1
            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.model_name=='resnet152':
            self.net = resnet.resnet152(pretrained=True)
            dim=1000
            hidden_dim = 128
            num_classes = 2
            dropout = 0.1
            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.model_name=='densenet121':
            self.net = densenet.densenet121(pretrained=True)
            dim=1000
            hidden_dim = 128
            num_classes = 2
            dropout = 0.1
            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.model_name=='densenet161':
            self.net = densenet.densenet161(pretrained=True)
            dim=1000
            hidden_dim = 128
            num_classes = 2
            dropout = 0.1
            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.model_name=='densenet169':
            self.net1 = densenet.densenet169(pretrained=True)
            self.net2 = densenet.densenet169(pretrained=True)
            self.net3 = densenet.densenet169(pretrained=True)

            dim=1000
            hidden_dim1 = 128
            hidden_dim2 = 64
            num_classes = 2
            dropout = 0.1
            self.classifier1 = nn.Sequential(
                nn.Linear(dim, hidden_dim1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier2 = nn.Sequential(
                nn.Linear(dim, hidden_dim1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier3 = nn.Sequential(
                nn.Linear(dim, hidden_dim1),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.classifier_final = nn.Sequential(
                nn.Linear(hidden_dim1*3, hidden_dim2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim2, num_classes))

        elif self.model_name == 'densenet201':
            self.net = densenet.densenet201(pretrained=True)

            dim = 1000
            hidden_dim = 128

            hidden_dim1 = 128
            hidden_dim2 = 64
            num_classes = 2
            dropout = 0.1
            # self.classifier1 = nn.Sequential(
            #     nn.Linear(dim, hidden_dim1),
            #     nn.GELU(),
            #     nn.Dropout(dropout),
            # )
            #
            # self.classifier_final = nn.Sequential(
            #     nn.Linear(hidden_dim1, hidden_dim2),
            #     nn.GELU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(hidden_dim2, num_classes))

            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )


        # elif self.model_name=='densenet201':
        #     self.net1 = densenet.densenet201(pretrained=True)
        #     self.net2 = densenet.densenet201(pretrained=True)
        #     self.net3 = densenet.densenet201(pretrained=True)
        #
        #     dim=1000
        #     hidden_dim1 = 128
        #     hidden_dim2 = 64
        #     num_classes = 2
        #     dropout = 0.1
        #     self.classifier1 = nn.Sequential(
        #         nn.Linear(dim, hidden_dim1),
        #         nn.GELU(),
        #         nn.Dropout(dropout),
        #     )
        #     self.classifier2 = nn.Sequential(
        #         nn.Linear(dim, hidden_dim1),
        #         nn.GELU(),
        #         nn.Dropout(dropout),
        #     )
        #     self.classifier3 = nn.Sequential(
        #         nn.Linear(dim, hidden_dim1),
        #         nn.GELU(),
        #         nn.Dropout(dropout),
        #     )
        #
        #     self.classifier_final = nn.Sequential(
        #         nn.Linear(hidden_dim1*3, hidden_dim2),
        #         nn.GELU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(hidden_dim2, num_classes))

        elif self.model_name=='inception':
            self.net = inception.Inception3()
            dim=1000
            hidden_dim = 128
            num_classes = 1
            dropout = 0.1
            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )


        elif self.model_name=='cait':
            dim = 1000
            hidden_dim = 128
            num_classes = 1
            dropout = 0.1

            self.net = cait.CaiT(image_size=256,
                            patch_size=32,
                            num_classes=num_classes,
                            dim=dim,
                            depth=12,  # depth of transformer for patch to patch attention only
                            cls_depth=2,  # depth of cross attention of CLS tokens to patch
                            heads=16,
                            mlp_dim=2048,
                            dropout=dropout,
                            emb_dropout=0.1,
                            layer_dropout=0.05)  # randomly dropout 5% of the layers

            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        elif self.model_name=='cross_vit':
            sm_dim = 192
            lg_dim = 384
            dim = sm_dim + lg_dim
            hidden_dim = 128
            num_classes = 1
            dropout = 0.1

            self.net = cross_vit.CrossViT(image_size=256,
                                   num_classes=2,
                                   depth=4,  # number of multi-scale encoding blocks
                                   sm_dim=192,  # high res dimension
                                   sm_patch_size=16,  # high res patch size (should be smaller than lg_patch_size)
                                   sm_enc_depth=2,  # high res depth
                                   sm_enc_heads=8,  # high res heads
                                   sm_enc_mlp_dim=2048,  # high res feedforward dimension
                                   lg_dim=384,  # low res dimension
                                   lg_patch_size=64,  # low res patch size
                                   lg_enc_depth=3,  # low res depth
                                   lg_enc_heads=8,  # low res heads
                                   lg_enc_mlp_dim=2048,  # low res feedforward dimensions
                                   cross_attn_depth=2,  # cross attention rounds
                                   cross_attn_heads=8,  # cross attention heads
                                   dropout=dropout,
                                   emb_dropout=0.1
                                   )

            self.classifier = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )


        #for metric
        self.train_preds = []
        self.train_gts = []

        self.valid_preds = []
        self.valid_gts = []

        self.test_preds = []
        self.test_probs = []
        self.test_gts = []

        self.oids  = []
        self.unums = []

    def forward(self, x1):
        x1 = self.net(x1)
        x1 = self.classifier(x1)

        # x2 = self.net2(x2)
        # x2 = self.classifier2(x2)
        #
        # x3 = self.net3(x3)
        # x3 = self.classifier3(x3)
        #
        # x = torch.cat((x1,x2,x3), dim=1)
        # self.classifier_final(x)

        return x1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=self.l2_norm)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        #scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.98)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x1, y, _, _, _ = batch
        y_hat = self.forward(x1)
        # print(y_hat.shape)
        # print(y.shape)
        loss = F.cross_entropy(y_hat, y)

        norm = torch.FloatTensor([0]).cuda()
        for parameter in self.parameters():
            norm += torch.norm(parameter, p=1)
        loss = loss + self.l1_norm * norm

        for gy in y:
            self.train_gts.append(gy.cpu().detach().numpy())
        for py in y_hat:
            c = torch.argmax(py)
            self.train_preds.append(c.cpu().detach().numpy())

        self.log("loss", loss, on_epoch=True, prog_bar=True)

        return loss


    def training_epoch_end(self, outputs):
        acc, sen, spe, ppv, npv = self.calculate_metirc(self.train_gts, self.train_preds)
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log("train_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        self.log("train_sen", sen, on_epoch=True, prog_bar=True)
        self.log("train_spe", spe, on_epoch=True, prog_bar=True)
        self.log("train_ppv", ppv, on_epoch=True, prog_bar=True)
        self.log("train_npv", npv, on_epoch=True, prog_bar=True)

        self.train_preds = []
        self.train_gts = []

        with open(self.model_path + '/train_loss.txt', "a") as f:
            f.write('{} {} {} {}'.format(avg_loss, acc, sen, spe) + '\n')


    def validation_step(self, batch, batch_idx):
        x1, y, _, _, _ = batch
        y_hat = self.forward(x1)
        loss = F.cross_entropy(y_hat, y)

        for gy in y:
            self.valid_gts.append(gy.cpu().detach().numpy())
        for py in y_hat:
            c = torch.argmax(py)
            self.valid_preds.append(c.cpu().detach().numpy())

        #print(len(self.valid_gts), len(self.valid_preds))

        acc, sen, spe, ppv, npv = self.calculate_metirc(self.valid_gts, self.valid_preds)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_sen", sen, on_epoch=True, prog_bar=True)
        self.log("val_spe", spe, on_epoch=True, prog_bar=True)
        self.log("val_ppv", ppv, on_epoch=True, prog_bar=True)
        self.log("val_npv", npv, on_epoch=True, prog_bar=True)
        self.log("val_bat_loss", loss, on_epoch=True, prog_bar=True)

        return {"val_bat_loss": loss, "val_acc": acc,
                "val_sen": sen, "va_spe": spe,
                "val_ppv": ppv, "val_npv": npv}

    def validation_epoch_end(self, outputs):
        #print("validation_epoch_end",len(self.valid_gts), len(self.valid_preds))
        acc, sen, spe, ppv, npv = self.calculate_metirc(self.valid_gts, self.valid_preds)
        avg_loss = torch.stack([x['val_bat_loss'] for x in outputs]).mean()

        self.log("val_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_sen", sen, on_epoch=True, prog_bar=True)
        self.log("val_spe", spe, on_epoch=True, prog_bar=True)
        self.log("val_ppv", ppv, on_epoch=True, prog_bar=True)
        self.log("val_npv", npv, on_epoch=True, prog_bar=True)

        self.valid_preds = []
        self.valid_gts = []

        with open(self.model_path + '/valid_loss.txt', "a") as f:
            f.write('{} {} {} {}'.format(avg_loss, acc, sen, spe) + '\n')

        return {"val_avg_loss": avg_loss, "val_acc": acc,
                "val_sen": sen, "va_spe": spe,
                "val_ppv":ppv, "val_npv": npv}

    def test_step(self, batch, batch_idx):
        x1, y, oids, unums, _ = batch
        y_hat = self.forward(x1)
        loss = F.cross_entropy(y_hat, y)

        for gy in y:
            self.test_gts.append(gy.cpu().detach().numpy())
        for py in y_hat:
            c = torch.argmax(py)
            p = F.softmax(py, dim=0)[1]
            self.test_probs.append(p.cpu().detach().numpy())
            self.test_preds.append(c.cpu().detach().numpy())

        for i in oids:
            self.oids.append(i.cpu().detach().numpy())

        for j in unums:
            self.unums.append(j.cpu().detach().numpy())

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        acc, sen, spe, ppv, npv = self.calculate_metirc(self.test_gts, self.test_preds)
        auc = self.calculate_auc(self.test_gts, self.test_probs)
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        gts, preds, probs = self.test_gts, self.test_preds, self.test_probs

        self.log("test_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        self.log("test_sensitivity(recall)", sen, on_epoch=True, prog_bar=True)
        self.log("test_specificity", spe, on_epoch=True, prog_bar=True)
        self.log("test_ppv(precision)", ppv, on_epoch=True, prog_bar=True)
        self.log("test_npv", npv, on_epoch=True, prog_bar=True)
        self.log("test_auc", auc, on_epoch=True, prog_bar=True)

        dfOID = pd.DataFrame(np.array(self.oids))
        dfUNUM = pd.DataFrame(np.array(self.unums))


        dfGTs = pd.DataFrame(np.round_(np.array(self.test_gts)))
        dfPreds = pd.DataFrame(np.round_(np.array(self.test_preds)))
        dfProbs = pd.DataFrame((np.array(self.test_probs)))


        pd.concat([dfOID, dfUNUM, dfGTs, dfPreds, dfProbs], axis=1).to_csv('./test.csv', index=False)

        return {"test_avg_loss": avg_loss, "test_acc": acc,
                "test_sensitivity(recall)": sen, "test_specificity": spe,
                "test_ppv(precision)": ppv, "test_npv": npv, "test_auc": auc}
                # "gts": gts, "pred_probs": probs, "preds": preds}

    def calculate_metirc(self, gts, preds):
        tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[0,1]).ravel()
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        acc = (tp + tn) / (tn + fp + fn + tp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)

        if math.isnan(sen):
            sen = 0
        if math.isnan(spe):
            spe = 0
        if math.isnan(ppv):
            ppv = 0
        if math.isnan(npv):
            npv = 0
        if math.isnan(acc):
            acc = 0

        return acc, sen, spe, ppv, npv

    def calculate_auc(self, gts, probs):
        try:
            auc = roc_auc_score(gts, probs)
        except ValueError:
            auc=0
        return auc