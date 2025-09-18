# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:40:44 2024

Analyze 2D Slices from files


@author: csarosiek
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


good = ['GT198',
'GT201',
'GT202',
'GT204',
'GT210',
'GT212',
'GT213',
'GT215',
'GT217',
'GT219',#]
#
#maybe = [
'GT205',
'GT207',
'GT208',
'GT216',
'GT218',
'GT221']

toppath = 'G:/Physicist/people/Sarosiek/1_DenseUNet_DelRec/multiorgan_data/organs_7/'

organs = ['Duodenum','Stomach','Colon','Bowel_Small','Liver','Kidney_R','Kidney_L']
folders = ['ACC-multiorgan-RR','ACC-multiorgan-Protege','ACC-multiorgan-Admire']

for folder in folders:
    o =1
    path = os.path.join(toppath,folder)

    files = [f for f in os.listdir(path) if '2D.csv' in f]

    files = [f for f in files if any(x in f for x in good)]

    df_list = (pd.read_csv(os.path.join(path,f)) for f in files)
    df_all = pd.concat(df_list, ignore_index=True)


    while o < 5:


        df_slices = df_all[(df_all['MDAInit']!=0) & (df_all['organ']==o)]

        TGInit = []
        TGACC = []
        for index, row in df_slices.iterrows():
            if row['DSCInit'] < 0.5 or row['MDAInit'] > 8:
                tgclass = 3
            else:
                tgclass = 1
            TGInit.append(tgclass)

            if row['DSCACC'] < 0.5 or row['MDAACC'] > 8:
                tgclass = 3
            else:
                tgclass = 1
            TGACC.append(tgclass)

        df_slices['TGInit'] = TGInit
        df_slices['TGACC'] = TGACC

        AcceptInit = df_slices[df_slices['TGInit']==1].count()['TGInit']
        #MinorInit = df_slices[df_slices['TGInit']==2].count()['TGInit']
        MajorInit = df_slices[df_slices['TGInit']==3].count()['TGInit']
        #ExtraInit = df_slices[df_slices['TGInit']==4].count()['TGInit']
        #MissingInit = df_slices[df_slices['TGInit']==5].count()['TGInit']

        AcceptACC = df_slices[df_slices['TGACC']==1].count()['TGACC']
        #MinorACC = df_slices[df_slices['TGACC']==2].count()['TGACC']
        MajorACC = df_slices[df_slices['TGACC']==3].count()['TGACC']
        #ExtraACC = df_slices[df_slices['TGACC']==4].count()['TGACC']
        #MissingACC = df_slices[df_slices['TGACC']==5].count()['TGACC']

        percentimprove = round(100 - (MajorACC/MajorInit*100),1)



        X = ['DLAS','ACC']
        Y1 = [AcceptInit, AcceptACC]
        #Y2= [MinorInit, MinorACC]
        #2 = [sum(x) for x in zip(Y1, Y2)]
        Y3 = [MajorInit, MajorACC]
        y3 = [sum(x) for x in zip(Y1, Y3)]
        # Y4 = [ExtraInit, ExtraACC]
        # y4 = [sum(x) for x in zip(Y1, Y2, Y3, Y4)]
        # Y5 = [MissingInit, MissingACC]
        plt.bar(X,Y1,label='Acceptable: '+str(AcceptInit)+' -> '+str(AcceptACC))
        #plt.bar(X,Y2,bottom=Y1,label='Minor')
        plt.bar(X,Y3,bottom=Y1,label='Major: '+str(MajorInit)+' -> '+str(MajorACC))
        #plt.bar(X,Y4,bottom=y3,label='Extra')
        #plt.bar(X,Y5,bottom=y4,label='Missing')
        plt.title(folder.split('-')[2]+': '+organs[o-1] +'\n Improved '+str(percentimprove)+'% of Major Slices')
        plt.legend(loc=8)
        plt.show()

        print(Y1)
        #print(Y2)
        print(Y3)

        o += 1




            # df_minor = df_all[(df_all['ClassInit']==2) & (df_all['organ']==o)].astype(float)
            # df_major = df_all[(df_all['ClassInit']==3) & (df_all['organ']==o)].astype(float)






            # print(o)
            # minor = df_minor.mean(axis=0)
            # print(minor)
            # print('\n')
            # major = df_major.mean(axis=0)
            # print(major)

            # df_minor.boxplot(column=['DSCInit','DSCACC'],meanline=True)
            # plt.title('Minor - DSC')
            # plt.show()

            # df_minor.boxplot(column=['SDSCInit','SDSCACC'],meanline=True)
            # plt.title('Minor - SDSC')
            # plt.show()

            # df_minor.boxplot(column=['MDAInit','MDAACC'],meanline=True)
            # plt.title('Minor - MDA')
            # plt.show()

            # df_minor.boxplot(column=['HD95Init','HD95ACC'],meanline=True)
            # plt.title('Minor - HD95')
            # plt.show()

            # df_minor.boxplot(column=['APLInit','APLACC'],meanline=True)
            # plt.title('Minor - APL')
            # plt.show()

            # df_major.boxplot(column=['DSCInit','DSCACC'],meanline=True)
            # plt.title('major - DSC')
            # plt.show()

            # df_major.boxplot(column=['SDSCInit','SDSCACC'],meanline=True)
            # plt.title('major - SDSC')
            # plt.show()

            # df_major.boxplot(column=['MDAInit','MDAACC'],meanline=True)
            # plt.title('major - MDA')
            # plt.show()

            # df_major.boxplot(column=['HD95Init','HD95ACC'],meanline=True)
            # plt.title('major - HD95')
            # plt.show()

            # df_major.boxplot(column=['APLInit','APLACC'],meanline=True)
            # plt.title('major - APL')
            # plt.show()


            # df_nonzero = df_all[(df_all['DSCInit']<1) & (df_all['organ']==o)]

            # AcceptInit = df_nonzero[df_nonzero['ClassInit']==1].count()['ClassInit']
            # MinorInit = df_nonzero[df_nonzero['ClassInit']==2].count()['ClassInit']
            # MajorInit = df_nonzero[df_nonzero['ClassInit']==3].count()['ClassInit']
            # ExtraInit = df_nonzero[df_nonzero['ClassInit']==4].count()['ClassInit']
            # MissingInit = df_nonzero[df_nonzero['ClassInit']==5].count()['ClassInit']

            # AcceptACC = df_nonzero[df_nonzero['ClassACC']==1].count()['ClassACC']
            # MinorACC = df_nonzero[df_nonzero['ClassACC']==2].count()['ClassACC']
            # MajorACC = df_nonzero[df_nonzero['ClassACC']==3].count()['ClassACC']
            # ExtraACC = df_nonzero[df_nonzero['ClassACC']==4].count()['ClassACC']
            # MissingACC = df_nonzero[df_nonzero['ClassACC']==5].count()['ClassACC']



            # X = ['DLAS','ACC']
            # Y1 = [AcceptInit, AcceptACC]
            # Y2= [MinorInit, MinorACC]
            # y2 = [sum(x) for x in zip(Y1, Y2)]
            # Y3 = [MajorInit, MajorACC]
            # y3 = [sum(x) for x in zip(Y1, Y2, Y3)]
            # Y4 = [ExtraInit, ExtraACC]
            # y4 = [sum(x) for x in zip(Y1, Y2, Y3, Y4)]
            # Y5 = [MissingInit, MissingACC]
            # plt.bar(X,Y1,label='Acceptable')
            # plt.bar(X,Y2,bottom=Y1,label='Minor')
            # plt.bar(X,Y3,bottom=y2,label='Major')
            # plt.bar(X,Y4,bottom=y3,label='Extra')
            # plt.bar(X,Y5,bottom=y4,label='Missing')
            # plt.legend()
            # plt.show()


            # conf = df_nonzero.groupby(['ClassInit', 'ClassACC']).size().unstack(fill_value=0)
            # # plt.pcolor(conf)
            # # plt.yticks(np.arange(0.5, len(conf.index), 1), conf.index)
            # # plt.xticks(np.arange(0.5, len(conf.index), 1), conf.index)
            # # plt.ylabel('ACC Classification')
            # # plt.xlabel('DLAS Classification')
            # # plt.show()

