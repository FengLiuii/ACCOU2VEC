# -*- codeing = utf-8 -*-
# @Time : 2021/11/17 15:11
# @ Author : LF
# @ File : P-R_Curve.py
# @ Software : PyCharm
import numpy as np
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import warnings
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.interpolate import make_interp_spline
warnings.filterwarnings("ignore", category=FutureWarning)

# cresci2015_userdata_file_path = 'C:/Users/LF/PycharmProjects/papercode/userdata.txt'
# cresci2015_output_emb_file_path_dbot = 'cresci_model_emb/cresci_2015_dbot2vec3.emb'
# cresci2015_output_emb_file_path_node2vec = 'cresci_model_emb/embeddings.emb'
# cresci2015_output_emb_file_path_struc2vec = 'cresci_model_emb/struc2vec-cresci2015.emb'
# cresci2015_output_emb_file_path_bot2vec = 'cresci_model_emb/cresci-2015.emb'
# cresci2015_output_emb_file_path_deepwalk = 'cresci_model_emb/deepwalk2.emb'
# cresci2015_emb_model_dbot2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_dbot)
# cresci2015_emb_model_node2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_node2vec)
# cresci2015_emb_model_struc2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_struc2vec)
# cresci2015_emb_model_bot2vec = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_bot2vec)
# cresci2015_emb_model_deepwalk = KeyedVectors.load_word2vec_format(cresci2015_output_emb_file_path_deepwalk)


twitter1_userdata_file_path = './twitter_model_emb/twitter_edges.txt'
twitter2_userdata_file_path ='C:/Users/LF/PycharmProjects/papercode/twitter_dataset/50k_node_label.txt'
twitter_output_emb_file_path_dbot = 'C:/Users/LF/PycharmProjects/papercode/twitter_dataset/50k_graph.emb'
twitter_output_emb_file_path_node2vec = 'twitter_model_emb/node2vec_12k_graph.emb'
twitter_output_emb_file_path_bot2vec = 'twitter_model_emb/bot2vec_12k_graph2.emb'
twitter_output_emb_file_path_deepwalk = 'twitter_model_emb/deepwakl_12k_graph.emb'
twitter_emb_model_dbot2vec = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_dbot)
twitter_emb_model_node2vec = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_node2vec)
twitter_emb_model_bot2vec = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_bot2vec)
twitter_emb_model_deepwalk = KeyedVectors.load_word2vec_format(twitter_output_emb_file_path_deepwalk)
id1_label1_dict = {}

#cresci_2015标签和节点
# with open(cresci2015_userdata_file_path, 'r', encoding='utf-8') as f:
# # with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         # splits = line.split(' ')
#         splits = line.split('\t')
#         user_id = splits[0]
#         label = splits[4].replace('\n', '')
#         id_label_dict.update({user_id: int(label)})


with open(twitter2_userdata_file_path, 'r', encoding='utf-8') as f:
# with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # splits = line.split(' ')
        splits = line.split(' ')
        user_id = splits[0]
        label = splits[1].replace('\n', '')
        id1_label1_dict.update({user_id: int(label)})
id2_label2_dict = {}
with open(twitter1_userdata_file_path, 'r', encoding='utf-8') as f:
# with open(twitter_userdata_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # splits = line.split(' ')
        splits = line.split(' ')
        user_id = splits[0]
        label = splits[1].replace('\n', '')
        id2_label2_dict.update({user_id: int(label)})

X1 = []
y1 = []
X2 = []
y2 = []
X3 = []
y3 = []
X4 = []
y4 = []
# X5 = []
# y5 = []
# a

for idx, key in enumerate(twitter_emb_model_dbot2vec.wv.vocab):
    emb_vector = twitter_emb_model_dbot2vec.wv[key]
    X1.append(emb_vector)
    y1.append(id1_label1_dict[key])
for idx, key in enumerate(twitter_emb_model_bot2vec.wv.vocab):
    emb_vector = twitter_emb_model_bot2vec.wv[key]
    X2.append(emb_vector)
    y2.append(id2_label2_dict[key])
for idx, key in enumerate(twitter_emb_model_node2vec.wv.vocab):
    emb_vector = twitter_emb_model_node2vec.wv[key]
    X3.append(emb_vector)
    y3.append(id2_label2_dict[key])
for idx, key in enumerate(twitter_emb_model_deepwalk.wv.vocab):
    emb_vector = twitter_emb_model_deepwalk.wv[key]
    X4.append(emb_vector)
    y4.append(id2_label2_dict[key])
# for idx, key in enumerate(cresci2015_emb_model_deepwalk.wv.vocab):
#     emb_vector = cresci2015_emb_model_deepwalk.wv[key]
#     X5.append(emb_vector)
#     y5.append(id_label_dict[key])
print(Counter(y1))
smo = SMOTE(random_state=42)

# X1_smo, y1_smo = smo.fit_resample(X1, y1)
# X2_smo, y2_smo = smo.fit_resample(X2, y2)
# X4_smo, y4_smo = smo.fit_resample(X4, y4)

train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y1, test_size=9/ 10, random_state=2)
train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y2, test_size=9/ 10, random_state=2)
train_x3, test_x3, train_y3, test_y3 = train_test_split(X3, y3, test_size=9/ 10, random_state=2)
train_x4, test_x4, train_y4, test_y4 = train_test_split(X4, y4, test_size=9/ 10, random_state=2)
# train_x5, test_x5, train_y5, test_y5 = train_test_split(X5, y5, test_size=9/ 10, random_state=2)


# classify
svm_classifier1 = svm.SVC(kernel='linear', C=1)
svm_classifier1.fit(train_x1, train_y1)
svm_classifier2 = svm.SVC(kernel='linear', C=1)
svm_classifier2.fit(train_x2, train_y2)
svm_classifier3 = svm.SVC(kernel='linear', C=1)
svm_classifier3.fit(train_x3, train_y3)
svm_classifier4 = svm.SVC(kernel='rbf', C=5)
svm_classifier4.fit(train_x4, train_y4)
# svm_classifier5 = svm.SVC(kernel='rbf', C=1)
# svm_classifier5.fit(train_x5, train_y5)

y1_score = svm_classifier1.decision_function(test_x1)
y2_score = svm_classifier2.decision_function(test_x2)
y3_score = svm_classifier3.decision_function(test_x3)
y4_score = svm_classifier4.decision_function(test_x4)
# y5_score = svm_classifier5.decision_function(test_x5)

# roc_auc
precision1, recall1, _ = precision_recall_curve(test_y1, y1_score)
precision2, recall2, _ = precision_recall_curve(test_y2, y2_score)
precision3, recall3, _ = precision_recall_curve(test_y3, y3_score)
precision4, recall4, _ = precision_recall_curve(test_y4, y4_score)
# precision5, recall5, _ = precision_recall_curve(test_y5, y5_score)
#print(list(precision1),len(list(precision1)))
# metrics.plot_confusion_matrix(svm_classifier, test_x, test_y)

#roc_auc curve
# fpr1,tpr1,threshholds = roc_curve(test_y1,svm_classifier.fit(train_x1, train_y1).decision_function(test_x1))
# roc_auc1 = auc(fpr1,tpr1)
# fpr2,tpr2,threshholds = roc_curve(test_y2,svm_classifier.fit(train_x2, train_y2).decision_function(test_x2))
# roc_auc2 = auc(fpr2,tpr2)
# fpr3,tpr3,threshholds = roc_curve(test_y3,svm_classifier.fit(train_x3, train_y3).decision_function(test_x3))
# roc_auc3 = auc(fpr3,tpr3)
# fpr4,tpr4,threshholds = roc_curve(test_y4,svm_classifier.fit(train_x4, train_y4).decision_function(test_x4))
# roc_auc4 = auc(fpr4,tpr4)
# fpr5,tpr5,threshholds = roc_curve(test_y5,svm_classifier.fit(train_x5, train_y5).decision_function(test_x5))
# roc_auc5 = auc(fpr5,tpr5)


fig = plt.figure(figsize=(6,4),dpi=300)

ax1 = fig.add_subplot(111)
# draw roc_auc curve
# ax1.plot(fpr1,tpr1,label='dbot2vec (area = %0.4f)'% roc_auc1)
# ax1.plot(fpr2,tpr2,label='node2vec (area = %0.4f)'% roc_auc2)
# ax1.plot(fpr3,tpr3,label='struc2vec (area = %0.4f)'% roc_auc3)
# ax1.plot(fpr4,tpr4,label='bot2vec (area = %0.4f)'% roc_auc4)
# ax1.plot(fpr5,tpr5,label='deepwalk (area = %0.4f)'% roc_auc5)
# draw p_r curve
print(len(recall2), len(precision2))
recall4 = list(recall4)
recall4_new = []
precision4_new = []
for i in range(0, len(recall4)):
    if i%30 == 0 or i<5:
        recall4_new.append(recall4[i])
        precision4_new.append(precision4[i])
print(precision4_new)

precision5=[0.34076330981398334, 0.34065757818765036, 0.34071222329162654, 0.34076688592972887, 0.34082156611039793,
            0.34176603287141477, 0.3427784974093264, 0.3436381386267491, 0.34499672988881624, 0.3463687150837989,
            0.3465984147952444, 0.3479920345170926, 0.34906604402935293, 0.3503184713375796, 0.35141509433962265,
            0.352353538774128, 0.35296119809394144, 0.3535750940814232, 0.3547111416781293, 0.35603180089872105,
            0.35684503127171646, 0.3581907090464548, 0.359375, 0.3603953406283092, 0.3616039744499645, 0.3630039243667499,
            0.36406025824964133, 0.36476739992787593, 0.366751269035533, 0.3678454247174626, 0.36950146627565983,
            0.37099152230003685, 0.37249814677538917, 0.37364890048453225, 0.37537481259370314, 0.3771202412363362,
            0.378316906747536, 0.3787647731605032, 0.3799846625766871, 0.38179714616274585, 0.3834367726920093,
            0.38451033944596175, 0.38540031397174257, 0.3866956178444532, 0.38780778395552024, 0.38973232121454254,
            0.3904742765273312, 0.39223615042458554, 0.39401952807160295, 0.39500613999181333, 0.39600494233937394,
            0.39825942809780357, 0.39929107589658047, 0.40138480906420476, 0.4028716216216216, 0.4048023799405015,
            0.4059024807527802, 0.4076625053809729, 0.40922876949740034, 0.41146969036197123, 0.41286215978928886,
            0.41493592576226246, 0.4165925266903915, 0.4180474697716077, 0.4188458070333634, 0.42056286881525196,
            0.4225319926873857, 0.42406810860561434, 0.4256255792400371, 0.42790480634624356, 0.4292763157894737,
            0.43090392806436345, 0.4325548141086749, 0.43350936149783964, 0.4347195357833656, 0.436678032148076,
            0.4381746810598626, 0.439693524468611, 0.4409860557768924, 0.4427997992975414, 0.44565217391304346,
            0.44727457972491086, 0.449435318275154, 0.45162959130884633, 0.45307612095933264, 0.4550709406200736,
            0.4568326271186441, 0.4594233849439402, 0.46178686759956944, 0.4639175257731959, 0.4666301969365427,
            0.4685603971318257, 0.4699666295884316, 0.4725182277061133, 0.47398190045248867, 0.47746719908727897,
            0.48072497123130037, 0.4820081253627394, 0.4844847775175644, 0.487005316007088, 0.4898688915375447,
            0.49128081779915816, 0.4933252427184466, 0.4947948560930802, 0.4978368355995056, 0.5, 0.5009445843828715,
            0.5044500953591863, 0.5077021822849808, 0.5100453661697991, 0.5111256544502618, 0.5122273628552545,
            0.514018691588785, 0.5171948752528658, 0.5194141689373297, 0.5220233998623538, 0.5250347705146036,
            0.5277582572030921, 0.5305397727272727, 0.5323043790380474, 0.5344702467343977, 0.5388848129126926,
            0.5408011869436202, 0.5435108777194299, 0.5443854324734446, 0.5479662317728319, 0.5516304347826086,
            0.5561665357423409, 0.5580286168521462, 0.5595333869670153, 0.5614820846905537, 0.5634789777411376,
            0.5663606010016694, 0.5710059171597633, 0.5732020547945206, 0.5767562879444926, 0.5804042179261862,
            0.585040071237756, 0.5857400722021661, 0.5873741994510522, 0.5904452690166976, 0.5945437441204139,
            0.601145038167939, 0.6045498547918683, 0.6075638506876228, 0.6121635094715853, 0.6143724696356275,
            0.6176772867420349, 0.622651356993737, 0.6272534464475079, 0.6330818965517241, 0.6369112814895947,
            0.6397550111358574, 0.6438278595696489, 0.6497695852534562, 0.6547479484173505, 0.6575178997613366,
            0.6640340218712029, 0.6676980198019802, 0.6702395964691047, 0.6767352185089974, 0.6795543905635649,
            0.6851604278074866, 0.6882673942701227, 0.6949860724233984, 0.6941678520625889, 0.6984011627906976,
            0.7035661218424963, 0.709726443768997, 0.7115085536547434, 0.7157643312101911, 0.7194127243066885,
            0.7249163879598662, 0.7307032590051458, 0.738556338028169, 0.740506329113924, 0.7462825278810409,
            0.7466539196940727, 0.7490157480314961, 0.7494929006085193, 0.7562761506276151, 0.7591792656587473,
            0.7689732142857143, 0.7713625866050808, 0.777511961722488, 0.78287841191067, 0.7809278350515464,
            0.7815013404825737, 0.7835195530726257, 0.7871720116618076, 0.7850609756097561, 0.7939297124600639,
            0.7969798657718121, 0.8256537102473498, 0.8322388059701493, 0.848300395256917, 0.8530252100840336,
            0.8651121076233184, 0.8717307692307693, 0.8867875647668394, 0.8914606741573034, 0.9096932515337423,
            0.917972972972973, 0.9221052631578947, 0.93305084745762712, 0.94300970873786407, 0.95181818181818182,
            0.9615068493150685, 0.97448275862068966, 0.98604651162790697, 0.9995294117647059, 1.0]

recall5 = [1.0, 0.9995294117647059, 0.9995294117647059, 0.9995294117647059, 0.9995294117647059, 0.9981176470588236,
            0.9962352941176471, 0.9938823529411764, 0.9929411764705882, 0.992, 0.987764705882353, 0.9868235294117647,
            0.9849411764705882, 0.9835294117647059, 0.9816470588235294, 0.9792941176470589, 0.976, 0.9727058823529412,
            0.9708235294117648, 0.9694117647058823, 0.9665882352941176, 0.9651764705882353, 0.9632941176470589, 0.9609411764705882,
            0.9590588235294117, 0.9576470588235294, 0.9552941176470588, 0.952, 0.952, 0.9496470588235294, 0.9487058823529412,
            0.9472941176470588, 0.9458823529411765, 0.9435294117647058, 0.9425882352941176, 0.9416470588235294, 0.9392941176470588,
            0.9350588235294117, 0.9327058823529412, 0.9317647058823529, 0.9303529411764706, 0.9275294117647059, 0.924235294117647,
            0.9218823529411765, 0.9190588235294118, 0.9181176470588235, 0.9143529411764706, 0.9129411764705883, 0.9115294117647059,
            0.908235294117647, 0.9049411764705882, 0.9044705882352941, 0.9011764705882352, 0.900235294117647, 0.8978823529411765,
            0.8964705882352941, 0.8931764705882353, 0.8912941176470588, 0.8889411764705882, 0.888, 0.8851764705882353, 0.8837647058823529,
            0.8814117647058823, 0.8785882352941177, 0.8743529411764706, 0.872, 0.8701176470588236, 0.8672941176470588, 0.8644705882352941,
            0.8630588235294118, 0.859764705882353, 0.8569411764705882, 0.8541176470588235, 0.8498823529411764, 0.8461176470588235,
            0.843764705882353, 0.8404705882352941, 0.8371764705882353, 0.8334117647058824, 0.8305882352941176, 0.8296470588235294,
            0.8263529411764706, 0.824, 0.8216470588235294, 0.8178823529411765, 0.8150588235294117, 0.8117647058823529, 0.8098823529411765,
            0.8075294117647058, 0.8047058823529412, 0.8028235294117647, 0.7995294117647059, 0.7952941176470588, 0.7929411764705883,
            0.7887058823529411, 0.7877647058823529, 0.7863529411764706, 0.7816470588235294, 0.7788235294117647, 0.776, 0.7736470588235295,
            0.7689411764705882, 0.7651764705882353, 0.7604705882352941, 0.7581176470588236, 0.7543529411764706, 0.7487058823529412,
            0.7468235294117647, 0.7444705882352941, 0.7407058823529412, 0.7350588235294118, 0.7294117647058823, 0.7247058823529412,
            0.7218823529411764, 0.7176470588235294, 0.7138823529411765, 0.7105882352941176, 0.7068235294117647, 0.7030588235294117,
            0.6978823529411765, 0.6931764705882353, 0.6912941176470588, 0.6861176470588235, 0.6818823529411765, 0.6752941176470588,
            0.672, 0.6687058823529411, 0.6663529411764706, 0.6607058823529411, 0.6545882352941177, 0.6489411764705882, 0.6432941176470588,
            0.6385882352941177, 0.6357647058823529, 0.6301176470588236, 0.6258823529411764, 0.6216470588235294, 0.6183529411764705, 0.6108235294117647,
            0.6042352941176471, 0.5990588235294118, 0.5948235294117648, 0.5929411764705882, 0.587764705882353, 0.5821176470588235, 0.5778823529411765,
            0.5712941176470588, 0.5656470588235294, 0.5614117647058824, 0.5567058823529412, 0.5529411764705883, 0.5472941176470588,
            0.5407058823529411, 0.5350588235294118, 0.5308235294117647, 0.5256470588235295, 0.5185882352941177, 0.5143529411764706,
            0.5077647058823529, 0.5002352941176471, 0.4955294117647059, 0.488, 0.4823529411764706, 0.4748235294117647,
            0.4696470588235294, 0.4592941176470588, 0.45223529411764707, 0.4456470588235294, 0.4395294117647059,
            0.43058823529411766, 0.42305882352941176, 0.41505882352941176, 0.408, 0.40094117647058825, 0.3948235294117647,
            0.38541176470588234, 0.37788235294117645, 0.3675294117647059, 0.35811764705882354, 0.3477647058823529,
            0.3402352941176471, 0.3308235294117647, 0.32423529411764707, 0.3143529411764706, 0.3058823529411765,
            0.2969411764705882, 0.2851764705882353, 0.2743529411764706, 0.264, 0.2541176470588235, 0.24235294117647058,
            0.23388235294117646, 0.2235294117647059, 0.21458823529411764, 0.2023529411764706, 0.19247058823529412,
            0.18211764705882352, 0.1731764705882353, 0.1628235294117647, 0.152, 0.13929411764705882, 0.1303529411764706,
            0.11811764705882354, 0.10541176470588236, 0.09223529411764705, 0.08047058823529411, 0.06776470588235294, 0.056,
            0.04611764705882353, 0.034823529411764705, 0.02258823529411765, 0.010823529411764706]




ax1.plot(recall2,precision2,'#00008B',label='Bot2vec',linewidth=1.5,linestyle='-' )
ax1.plot(recall3,precision3,'#FF8C00',label='Node2vec',linewidth=1.5,linestyle='-' )
ax1.plot(recall5, precision5,'#808080',label='Deepwalk' ,linewidth=1.5,linestyle='-')
ax1.plot(recall1,precision1,'#FF0000',label='Accou2vec',linewidth=1.5,linestyle='-' )


print(len(recall3), len(precision3))
print(len(recall4), len(precision4))
print(len(recall1), len(precision1))
# ax1.plot(recall5,precision5,'g',label='Deepwalk' )

ax1.legend(loc='best',  prop={'family':'Times New Roman', 'size':8}, frameon=True)
plt.xlabel('Recall',fontdict={'family' : 'Times New Roman', 'size':12})
plt.ylabel('Precision',fontdict={'family' : 'Times New Roman', 'size':12})
# plt.title('The Precision-Recall Curve of Diffent Model in Twitter Data Set')
plt.savefig('C:/Users/LF/PycharmProjects/papercode/Figure/p-r-twitter.png',dpi=300, bbox_inches="tight")
plt.show()