import os
from plotnine import *
from plotnine_prism import * 
import pandas as pd
from pandas.api.types import CategoricalDtype
from globals import common_param, plot_param

save_type = plot_param['save_type']
columns = plot_param['columns']
dis_pal = plot_param['dis_pal']
con_pal = plot_param['con_pal']
tabs = plot_param['tabs']
figs = plot_param['figs']
save_it =  plot_param['save_it']
bg = common_param['budget_scalings']


def check_and_make_directories(directories: list[str]):
    print(f"Checking directories {directories}")
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)
        else:
            print(f"Directory `{directory}` already exist")
            
def wide_to_long(df):
    result_df = pd.melt(df, id_vars=plot_param['id_vars'], value_vars=plot_param['value_vars'], var_name='metric', value_name='value')
    result_df = result_df[plot_param['id_vars'] + ['metric', 'value']]
    result_df = result_df.sort_values(by=plot_param['id_vars']).reset_index().drop(columns='index')
    return result_df

def prepare_before_plot(data, pic_ordered):
    df = wide_to_long(data)
    bg0 = df['budget'].unique()
    df['budget'] = df['budget'].replace({x:y for x,y in zip(bg0,bg)})   # 原来的float 替换成 str
    budget_order = CategoricalDtype(bg, ordered=True)   # 排序
    df['budget'] = df['budget'].astype(budget_order)    
    ### 定义顺序方便作图和表
    method_order = CategoricalDtype(pic_ordered, ordered=True)
    df['method'] = df['method'].astype(method_order)
    df = df.dropna()
    df[df.metric=='ecpc']['value'] = 1/df[df.metric=='ecpc']['value']
    # 计算分组后的排序，注意必须是分组后再排序，应该按降序排列，value越大rank越小
    df['rank'] = df.groupby(['dataset', 'metric','budget','mask'])['value'].rank(ascending=False).astype(int)

    return df

def dataset_metric_matrix(d,data_name=None):
    # ggplot(ds)定义基于某个数据集作图
    # aes() 定义了若干映射：x轴由budget定义，y轴由mask定义，边缘颜色和填充颜色均由method的不同来自动分配
    # options(repr.plot.width =9, repr.plot.height =9)
    p = (ggplot(d, aes(x='budget', y='factor(mask)', color = 'rank', fill='rank')) # 定义映射
    + geom_tile(aes(width=0.95, height=.95), show_legend=False)   # heatmap
    + geom_text(aes(label='rank'), size=9, color="w") # heatmap上叠加文字
    # + geom_text(aes(label=c),size=6,color="black",nudge_x=.2, nudge_y=df[c].max()/50)
    # + scale_color_brewer(type="qual",palette='Set1') 
    # + scale_fill_brewer(type="qual", palette='Set1') 
    # + scale_color_prism('warm_and_sunny')
    # + scale_fill_prism('warm_and_sunny') 
    # + scale_color_manual(['white', 'black']) # new
    # + prism_fill_pal("autumn_leaves")
    # + theme_prism()
    + facet_grid("dataset~metric")
    # + ggsave("test.pdf")
    + ylab("mask")
    # + scale_fill_distiller(type='seq',palette='inferno')
    + scale_color_cmap(name= con_pal)     # plt.cm.ListedCmap
    + scale_fill_cmap(name= con_pal)     # plt.cm.ListedCmap
    + theme_minimal()
    )
    
    return p.save(figs +data_name+"ranking_detail"+ save_type, dpi=600, width=8, height=4,limitsize=False)  if save_it else p
# ggsave("test.pdf", units="in", dpi=300, width=8, height=4, device="pdf")


def cloud_rain(d,obj, data_name=None):
    # ggplot(ds)定义基于某个数据集作图
    # aes() 定义了若干映射：x轴由budget定义，y轴由mask定义，边缘颜色和填充颜色均由method的不同来自动分配
    # options(repr.plot.width =9, repr.plot.height =9)
    p = (ggplot(d, aes(x='method', y='rank', color = 'method', fill='method')) # 定义映射
        + geom_violin(style='right', alpha = 1, width = 2, show_legend=True, position = position_nudge(x = 0.0, y = 0))   # heatmap
        # + geom_point(position = 'jitter', size = 0.5, alpha = 0.5)
        + geom_jitter(size = 0.1, alpha = 0.5)
        # + geom_boxplot(style="left")   
        # + geom_text(aes(label='rank'), size=9, color="w") # heatmap上叠加文字
        # + geom_text(aes(label=c),size=6,color="black",nudge_x=.2, nudge_y=df[c].max()/50)
        # + scale_color_brewer(type="qual",palette='Set1') 
        # + scale_fill_brewer(type="qual", palette='Set1') 
        # + scale_color_prism('warm_and_sunny')
        # + scale_fill_prism('warm_and_sunny') 
        # + scale_color_manual(['white', 'black']) # new
        # + prism_fill_pal("autumn_leaves")
        # + facet_grid("dataset~metric")
        # + ggsave("test.pdf", units="in", dpi=300, width=8, height=4, device="pdf")
        + ylab("Ranking")
        + scale_color_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
        + scale_fill_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
        + theme_prism()
        # + theme_matplotlib()
        + scale_y_discrete(limits=range(1, len(obj)+1))
        + coord_flip()
        )
    return p.save(figs+"highlight"+save_type, dpi=600) if save_it else p

def dataset_metric_matrix(d,data_name=None):
    # ggplot(ds)定义基于某个数据集作图
    # aes() 定义了若干映射：x轴由budget定义，y轴由mask定义，边缘颜色和填充颜色均由method的不同来自动分配
    # options(repr.plot.width =9, repr.plot.height =9)
    p = (ggplot(d, aes(x='budget', y='factor(mask)', color = 'rank', fill='rank')) # 定义映射
    + geom_tile(aes(width=0.95, height=.95), show_legend=False)   # heatmap
    + geom_text(aes(label='rank'), size=9, color="w") # heatmap上叠加文字
    # + geom_text(aes(label=c),size=6,color="black",nudge_x=.2, nudge_y=df[c].max()/50)
    # + scale_color_brewer(type="qual",palette='Set1') 
    # + scale_fill_brewer(type="qual", palette='Set1') 
    # + scale_color_prism('warm_and_sunny')
    # + scale_fill_prism('warm_and_sunny') 
    # + scale_color_manual(['white', 'black']) # new
    + theme_prism()
    # + prism_fill_pal("autumn_leaves")
    # + theme_prism()
    # + ggsave("test.pdf")
    + ylab("mask")
    + ggtitle(data_name)
    + scale_color_cmap(name= con_pal)     # plt.cm.ListedCmap
    + scale_fill_cmap(name= con_pal)     # plt.cm.ListedCmap
    )
    return p.save(figs+data_name+" ranking"+save_type, dpi=600, width=8, height=4,limitsize=False) if save_it else p

# ggsave("test.pdf", units="in", dpi=300, width=8, height=4, device="pdf")

def method_boxplot(d,metric,data_name=None):
    # ggplot(ds)定义基于某个数据集作图
    # aes() 定义了若干映射：x轴由budget定义，y轴由mask定义，边缘颜色和填充颜色均由method的不同来自动分配
    # options(repr.plot.width =9, repr.plot.height =9)
    p = (ggplot(d, aes(x='method', y="value", fill='method')) # 定义映射
    + geom_boxplot(color = "k", alpha=1) #, position = position_nudge(x = 0.1, y = 0))#, show_legend=False)
    # + geom_violin(color = "k", alpha=1, style="right", position = position_nudge(x = -0.1, y = 0))#, show_legend=False)
    # + geom_point(size=1, color = "k", alpha=0.3)
    # + geom_jitter(alpha = 0.3)
    # + stat_summary() # fun_data = mean_se,median_hilow,mean_sdl,mean_cl_boot,mean_cl_normal
    # + geom_tile(aes(width=0.95, height=.95))   # heatmap
    # + geom_text(aes(label='rank'), size=9, color="w") # heatmap上叠加文字
    # + geom_text(aes(label=c),size=6,color="black",nudge_x=.2, nudge_y=df[c].max()/50)
    # + scale_color_brewer(palette='Blues') 
    # + scale_fill_brewer(palette='Blues') 
    + scale_color_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    + scale_fill_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    # + scale_color_prism('warm_and_sunny') 
    # + scale_fill_prism('warm_and_sunny') 
    # + scale_color_manual(['white', 'black']) # new
    # + theme(legend_position=(-0.3,-0.5))
    # + theme(axis_text_x=element_text(angle=90))
    # + prism_fill_pal("autumn_leaves")
    + theme_prism()
    + ylab(metric)
    # + theme(legend_position=(0.6,0.9), legend_direction='horizontal')
    # + coord_flip() 
    # + facet_grid("dataset~metric", scales="free")
    # + ggsave("test.pdf")
    # + coord_flip()
    )
    if data_name != None:
        return p.save(figs+data_name+metric+"_method_boxplot"+save_type) if save_it else p

def budget_line(d,yl,data_name=None):
    # ggplot(ds)定义基于某个数据集作图
    # aes() 定义了若干映射：x轴由budget定义，y轴由mask定义，边缘颜色和填充颜色均由method的不同来自动分配
    # options(repr.plot.width =9, repr.plot.height =9)
    p = (ggplot(d, aes(x='budget', y='value', color="method", fill='method',group="method",shape="method"))  # 定义映射
    + geom_line(size=1.5)
    # + geom_line(size=1)
    # + geom_boxplot()
    + geom_point(size=6, color = "k")    
    # + stat_summary(fun_data="mean_se")
    # + geom_tile(aes(width=0.95, height=.95))   # heatmap
    # + geom_text(aes(label='rank'), size=9, color="w") # heatmap上叠加文字
    # + geom_text(aes(label=c),size=6,color="black",nudge_x=.2, nudge_y=df[c].max()/50)
    + scale_color_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    + scale_fill_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    # + theme(legend_position=(-0.3,-0.5))
    # + theme(axis_text_x=element_text(angle=90))
    + theme_prism() # 使用prism的theme
    + ylab(yl)
    # + coord_flip()  # 反转坐标轴
    # + facet_grid("dataset~metric", scales="free")
    # + ggsave("test.pdf")  # 保存
    )
    return p.save(figs+data_name+yl+"_budget_line"+save_type) if save_it else p

def mask_line(d,yl,data_name=None):
    # ggplot(ds)定义基于某个数据集作图
    # aes() 定义了若干映射：x轴由budget定义，y轴由mask定义，边缘颜色和填充颜色均由method的不同来自动分配
    # options(repr.plot.width =9, repr.plot.height =9)
    p = (ggplot(d, aes(x='mask', y='value', color="method", fill='method',group="method",shape="method"))  # 定义映射
    + geom_line(size=1.5)
    + geom_point(size=6, color="k")    
    # + stat_summary(fun_data="mean_se")
    # + geom_tile(aes(width=0.95, height=.95))   # heatmap
    # + geom_text(aes(label='rank'), size=9, color="w") # heatmap上叠加文字
    # + geom_text(aes(label=c),size=6,color="black",nudge_x=.2, nudge_y=df[c].max()/50)
    + scale_color_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    + scale_fill_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    # + theme(legend_position=(-0.3,-0.5))
    # + theme(axis_text_x=element_text(angle=90))
    + theme_prism() # 使用prism的theme
    + ylab(yl)
    # + coord_flip()  # 反转坐标轴
    # + facet_grid("dataset~metric", scales="free")
    # + ggsave("test.pdf")  # 保存
    )
    return p.save(figs+data_name+yl+"_mask_line"+save_type) if save_it else p


def dataset_metric_matrix(d,data_name=None):
    # ggplot(ds)定义基于某个数据集作图
    # aes() 定义了若干映射：x轴由budget定义，y轴由mask定义，边缘颜色和填充颜色均由method的不同来自动分配
    # options(repr.plot.width =9, repr.plot.height =9)
    p = (ggplot(d, aes(x='budget', y='factor(mask)', color = 'rank', fill='rank')) # 定义映射
    + geom_tile(aes(width=0.95, height=.95), show_legend=False)   # heatmap
    + geom_text(aes(label='rank'), size=9, color="w") # heatmap上叠加文字
    # + geom_text(aes(label=c),size=6,color="black",nudge_x=.2, nudge_y=df[c].max()/50)
    # + scale_color_brewer(type="qual",palette='Set1') 
    # + scale_fill_brewer(type="qual", palette='Set1') 
    + scale_color_cmap(name= "Greens")     # plt.cm.ListedCmap
    + scale_fill_cmap(name= "Greens")     # plt.cm.ListedCmap
    # + scale_color_prism('warm_and_sunny')
    # + scale_fill_prism('warm_and_sunny') 
    # + scale_color_manual(['white', 'black']) # new
    + theme_void()
    # + prism_fill_pal("autumn_leaves")
    # + theme_prism()
    + facet_grid("dataset~metric")
    # + ggsave("test.pdf")
    + ylab("mask")
    )
    if data_name != None:
        return p.save(figs+data_name+"ranking"+save_type, dpi=600, width=8, height=4,limitsize=False)  if save_it else p
# ggsave("test.pdf", units="in", dpi=300, width=8, height=4, device="pdf")

def budget_bar(d,metric,data_name=None):
  p = (ggplot(d, aes(x = 'budget', y = 'value', fill = 'method')) 
    + geom_boxplot(size=0.5)
    # + geom_point(size=6, color="k")    
    + scale_color_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    + scale_fill_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    + theme_prism()
    + ylab(metric)
    # + theme(legend_text = element_text(size = 15))
    # + theme(axis_text_x = element_text(size=15))
    # + theme(axis_title_x = element_text(size=15))
    # + theme(axis_title_y = element_text(size=15))
    )
  if data_name != None:
    return p.save(figs+data_name+metric+"_budget_bar"+save_type) if save_it else p

def mask_bar(d,metric,data_name=None):
  p = (ggplot(d, aes(x = 'mask', y = 'value', fill = 'method')) 
    + geom_boxplot(size=0.5)
    # + geom_point(size=6, color="k")    
    + scale_color_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    + scale_fill_cmap_d(name= dis_pal)     # plt.cm.ListedCmap
    + theme_prism()
    + ylab(metric)
    # + theme(legend_text = element_text(size = 15))
    # + theme(axis_text_x = element_text(size=15))
    # + theme(axis_title_x = element_text(size=15))
    # + theme(axis_title_y = element_text(size=15))
    )
  if data_name != None:
    return p.save(figs+data_name+metric+"_mask_bar"+save_type) if save_it else p



def tabel_to_tex(df, tab_param={'tab_name' : ['mask'], 
                 'tabel_index' : ["method",'dataset'], 
                 'tabel_columns':['metric',"budget"], 
                 'tabel_values':['value'],
                 'float_format':"%.2f",
                 'position':"htbp",
                 'column_format': "l|l|cc|cc|cc|cc|cc|cc|cc|cc|cc|cc"}):
    grouby_var = tab_param['tab_name']
    for d in df.groupby(grouby_var): #, columns=['budget','mask','method'], values='value')
        dd = d[1][tab_param['tabel_index'] + tab_param['tabel_columns'] + tab_param['tabel_values']]
        dd = dd.round(3) # 小数点后3位够了
        # dd = dd.round({'value': 3}) # 小数点后3位够了
        tabel = dd.pivot_table(index=tab_param['tabel_index'], columns=tab_param['tabel_columns'], values=tab_param['tabel_values'])
        masked = float(d[1]['mask'].iloc[0])
        percent = "{:.2f}".format(masked)
        tabel.to_latex(
            buf=tabs+"{}_".format(tab_param['tab_name']) +percent+".tex",    # 文件名有标识
            caption="Comparative performance with {}\% masked".format(int(masked*100)), # 标题
            label="mask_"+percent,  # label
            float_format = tab_param['float_format'],    # 表格中的float，小数点后2位数
            position = tab_param['position'],        # 表格的位置
            column_format = tab_param['column_format'], # 表格列的对齐方式
            escape = True           # 对latex敏感的字段做escape处理，即在前面加"\"
            )
        
def plot_and_tex(df, pic_ordered, pic_category=['cloud_rain', 'heatmap', 'boxplot', 'budget_lineplot', 
                            'mask_lineplot', 'budget_barplot', 'mask_barplot'], 
                     tab_param={'tab_name' : ['mask'], 
                            'tabel_index' : ["method",'dataset'], 
                            'tabel_columns':['metric',"budget"], 
                            'tabel_values':['value'],
                            'float_format':"%.2f",
                            'position':"htbp",
                            'column_format': "l|l|cc|cc|cc|cc|cc|cc|cc|cc|cc|cc"}):
    check_and_make_directories([tabs, figs])
    df = prepare_before_plot(df, pic_ordered)
    if 'cloud_train' in pic_category:
        cloud_rain(df, pic_ordered)
    if 'heatmap' in pic_category:
        dataset_metric_matrix(df)
    if 'boxplot' in pic_category:
        [method_boxplot(df[df.method == "IIBidder"],dx0[0]+" on "+dx0[1]) for (dx0, dx1) in df.groupby(['metric','dataset'])]
    if 'budget_lineplot' in pic_category:
        for dx in df.groupby(['metric','dataset']):
            dx1 = dx[1].groupby(['budget','method']).value.median().reset_index()
            budget_line(dx1,dx[0][0]+" on "+ dx[0][1],'')
    if 'mask_lineplot' in pic_category:
        for dx in df.groupby(['metric','dataset']):
            dx1 = dx[1].groupby(['mask','method']).value.median().reset_index()
            mask_line(dx1, dx[0][0]+" on "+ dx[0][1],'')
    if 'budget_barplot' in pic_category:
        for dx in df.groupby(['metric','dataset']):
            dx1 = dx[1].groupby(['budget','method']).value.median().reset_index()
            budget_bar(dx1, dx[0][0]+" on "+ dx[0][1],'')
    if 'mask_barplot' in pic_category:
        for dx in df.groupby(['metric','dataset']):
            dx1 = dx[1].groupby(['mask','method']).value.median().reset_index()
            mask_bar(dx1, dx[0][0]+" on "+ dx[0][1],'')
    
    tabel_to_tex(df, tab_param)


if __name__ == '__main__':
    # data = pd.read_csv('C://Users\win10\Desktop//final.csv')
    data = pd.read_excel('final.xlsx')
    abl = plot_param['ablation']
    comp = plot_param['compared']
    abl_category = ['boxplot']
    comp_category = ['cloud_rain', 'heatmap', 'budget_lineplot', 
                            'mask_lineplot', 'budget_barplot', 'mask_barplot']
    plot_and_tex(data, abl, abl_category)
    # plot_and_tex(data, comp, comp_category)