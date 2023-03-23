"""
bmp文件由四部分构成，
1，位图文件头 bitmap-file header（该部分一共有14个字节，可以提供文件的格式，大小等信息）
        typedef struct tagBITMAPFILEHEADER
            {
            UINT16 bfType;    // 说明位图类型  2字节
            DWORD bfSize;  // 说明位图大小  4字节
            UINT16 bfReserved1;  // 保留字，必须为0  2字节
            UINT16 bfReserved2;  // 保留字，必须为0   2字节
            DWORD bfOffBits; // 从文件头到实际的图像数据的偏移量是多少  4字节
            } BITMAPFILEHEADER;  //一共14个字节

2，位图信息头（bitmap-information header）作为真彩色位图，我们主要关注biwidth 和 biheight 这两个数值，连个数值告诉我们图像的尺寸
        typedef struct tagBITMAPINFOHEADER
             {
            DWORD biSize;  // 说明该结构一共需要的字节数 2字节
            LONG biWidth;  // 说明图片的宽度，以像素为单位 4字节
            LONG biHeight; // 说明图片的高度，以像素为单位 4字节
            WORD biPlanes; //颜色板，总是设为1  2个字节
            WORD biBitCount;  //说明每个比特占多少bit位，可以通过这个字段知道图片类型  2个字节
            DWORD biCompression;  // 说明使用的压缩算法 2个字节 （BMP无压缩算法）
            DWORD biSizeImage;  //说明图像大小   2个字节
            LONG biXPelsPerMeter;  //水平分辨率 4字节  单位：像素/米
            LONG biYPelsPerMeter;  //垂直分辨率4字节
            DWORD biClrUsed;  //说明位图使用的颜色索引数 4字节
            DWORD biClrImportant; //4字节
            } BITMAPINFOHEADER; // 一共40个字节

3，颜色表 color table
    紧跟在位图信息头后面的就是数据就是颜色表了，颜色表是形如一个二维数组，4个字节为一行，
    这四个字节分别代表了R、G、B ，Alpha(透明度通道)的分量。通过位图数据，我们就可以对这像素进行着色。
    注意，如果位图是24位真彩色图像，那么就没有颜色表。

4， 位图点阵数据(bits datasets)
    在颜色表后面的就是像素点数据了，如果bmp是伪彩色图的话，那么每个像素只占8位，即一个字节。
    如果bmp是24位真彩色图像的话，那么每个像素占24位，即3个字节，3个字节分别为B、G、R颜色分量。
"""

# bad_miss.bmp   1.57MB  1646566 字节
# 像素数：1401*1172  1641972
# 像素所占字节 == 像素数量 == 1641972 个字节
# 文件信息头  14 个字节
# 位图信息头  40个字节
# 颜色表  256个颜色*4 = 1024个字节  （8位）伪彩色，每个像素占8位，也可以理解位gray，而RGB占24位

# 理论大小：1641972 + 14 +40 +1024 = 1643050
# 差值 ：1646566 - 1643050 = 3516 个字节
# （Windows系统中有“补零”的习惯(这是因为windows习惯4个字节扫描一次)，就是每行的像素宽度必须是4的倍数，
# 如果不能被4整除，则添加若干个0像素，直到每行像素能被4整除为止。）
# 而 1401 不能被 4 整除  需要补充3个像素，（也就是3个字节），所以最后的补值： 3*1172= 3516 个字节
#

# 类型: b'BM' 大小: 1646566 位图数据偏移量: 1078 宽度: 1401 高度: 1172 位图: 8

import numpy as np
import struct
import matplotlib.pyplot as plt


def main():
    '先将位图打开'
    f = open(r"D:\workdata\code\mytest\vpro_operator\beadinspect\bead\bad_miss.bmp", 'rb')  # 打开对应的文件
    '下面部分用来读取BMP位图的基础信息'
    f_type = str(f.read(2))  # 这个就可以用来读取 文件类型 需要读取2个字节
    file_size_byte = f.read(4)  # 这个可以用来读取文件的大小 需要读取4个字节
    f.seek(f.tell() + 4)  # 跳过中间无用的四个字节
    file_ofset_byte = f.read(4)  # 读取位图数据的偏移量
    f.seek(f.tell() + 4)  # 跳过无用的两个字节
    file_wide_byte = f.read(4)  # 读取宽度字节
    file_height_byte = f.read(4)  # 读取高度字节
    f.seek(f.tell() + 2)  ## 跳过中间无用的两个字节
    file_bitcount_byte = f.read(4)  # 得到每个像素占位大小

    # 下面就是将读取的字节转换成指定的类型
    f_size, = struct.unpack('l', file_size_byte)
    f_ofset, = struct.unpack('l', file_ofset_byte)
    f_wide, = struct.unpack('l', file_wide_byte)
    f_height, = struct.unpack('l', file_height_byte)
    f_bitcount, = struct.unpack('i', file_bitcount_byte)
    print("类型:", f_type, "大小:", f_size, "位图数据偏移量:", f_ofset, "宽度:", f_wide, "高度:", f_height, "位图:",
          f_bitcount)

    '然后来读取颜色表'
    color_table = np.empty(shape=[256, 4], dtype=int)
    f.seek(54)  # 跳过文件信息头和位图信息头
    for i in range(0, 256):
        b = struct.unpack('B', f.read(1))[0]
        g = struct.unpack('B', f.read(1))[0]
        r = struct.unpack('B', f.read(1))[0]
        alpha = struct.unpack('B', f.read(1))[0]
        color_table[i][0] = r
        color_table[i][1] = g
        color_table[i][2] = b
        color_table[i][3] = 255

    '下面部分用来读取BMP位图数据区域,将数据存入numpy数组'
    # 首先对文件指针进行偏移
    f.seek(f_ofset)
    # 因为图像是8位伪彩色图像，所以一个像素点占一个字节，即8位
    img = np.empty(shape=[f_height, f_wide, 4], dtype=int)
    cout = 0
    for y in range(0, f_height):
        for x in range(0, f_wide):
            cout = cout + 1
            index = struct.unpack('B', f.read(1))[0]
            img[f_height - y - 1, x] = color_table[index]
        while cout % 4 != 0:
            f.read(1)
            cout = cout + 1
    plt.imshow(img)
    plt.show()
    f.close()


if __name__ == '__main__':
    main()
