import numpy as np
import tensorflow as tf


def sigmoid(a):
    return 1. / (1. + np.exp(-a))

#
# def dense_layers(input_tensor, out_dim, name, norm_rate=0.0, activation=None, bias=False):
#     regularizer = tf.contrib.layers.l2_regularizer(norm_rate)
#
#     outs = tf.layers.dense(input_tensor, out_dim, activation=activation, kernel_regularizer=regularizer,
#                            reuse=tf.AUTO_REUSE, use_bias=bias, name=name)
#     return outs


from PyPDF2 import PdfFileReader,PdfFileWriter
def MergePDF(filepath, outfile):

    output = PdfFileWriter()
    outputPages = 0
    for name in ['chapter{}.pdf'.format(i) for i in range(1, 16)] + ['']

    if pdf_fileName:
        for pdf_file in pdf_fileName:
            print("路径：%s"%pdf_file)

            # 读取源PDF文件
            input = PdfFileReader(open(pdf_file, "rb"))

            # 获得源PDF文件中页面总数
            pageCount = input.getNumPages()
            outputPages += pageCount
            print("页数：%d"%pageCount)

            # 分别将page添加到输出output中
            for iPage in range(pageCount):
                output.addPage(input.getPage(iPage))

        print("合并后的总页数:%d."%outputPages)
        # 写入到目标PDF文件
        outputStream = open(os.path.join(filepath, outfile), "wb")
        output.write(outputStream)
        outputStream.close()
        print("PDF文件合并完成！")

    else:
        print("没有可以合并的PDF文件！")