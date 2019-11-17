import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def labelBarGraph(label):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 8
    plt.rcParams["figure.figsize"] = fig_size
    label.sum(axis=0).plot.bar()
    plt.savefig("Visualizations/Label_Bar_Graph.png")
    plt.show()

def lengthHistogram(comment):
    x = [len(comment[i]) for i in range(comment.shape[0])]
    print('average length of comment: {:.3f}'.format(sum(x)/len(x)) )
    bins = [1,200,400,600,800,1000,1200]
    plt.hist(x, bins=bins)
    plt.xlabel('Length of comments')
    plt.ylabel('Number of comments')       
    plt.axis([0, 1200, 0, 90000])
    plt.grid(True)
    plt.savefig("Visualizations/Length_Of_Comments_Histogram.png", bbox_inches='tight')
    plt.show()

def classifiedCommentsLengthHistogram(comment,label):
    label = label.values
    y = np.zeros(label.shape)
    for i in range(comment.shape[0]):
        l = len(comment[i])
        if label[i][0] :
            y[i][0] = l
        if label[i][1] :
            y[i][1] = l
        if label[i][2] :
            y[i][2] = l
        if label[i][3] :
            y[i][3] = l
        if label[i][4] :
            y[i][4] = l
        if label[i][5] :
            y[i][5] = l

    labelsplt = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    color = ['red','green','blue','yellow','orange','chartreuse']
    bins = [1,200,400,600,800,1000,1200]
    plt.hist(y,bins = bins, label = labelsplt,color = color)
    plt.axis([0, 1200, 0, 8000])
    plt.xlabel('Length of comments')
    plt.ylabel('Number of comments')
    plt.legend()
    plt.grid(True)
    plt.savefig("Visualizations/Classified_Comments_Histogram.png")
    plt.show()


def CorrMatrix(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(12,10))
    matrix = sns.heatmap(corr_matrix,annot = True)
    matrix =  matrix.get_figure()
    matrix.savefig('./Visualizations/Correlation_Matrix.png')


if __name__ == "__main__":
    data = pd.read_csv("Data/train.csv")
    comment = data["comment_text"]
    label = data[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]

    #labelBarGraph(label)
    #lengthHistogram(comment)
    #classifiedCommentsLengthHistogram(comment,label)
    CorrMatrix(data)
