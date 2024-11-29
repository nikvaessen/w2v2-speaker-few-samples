# install the tidyverse
library(tidyverse)
library(patchwork)
library(scales)
library(ggthemes)
library(egg)
library(extrafont)

# font_install('fontcm')
font_import()
loadfonts()

# read data
df = read.csv('plot_four.csv')

# clip NAN results
df$eer <- ifelse(df$eer > 0.50, 0.50, df$eer)

# change legend names
df$network[df$network == "xvector"] <- "X-vector"
df$network[df$network == "ecapa"] <- "ECAPA-TDNN"
df$network[df$network == "wav2vec2"] <- "wav2vec 2.0"

# subdataframes for each task
df_vox2 = df[df$data == "vox2",]
df_spk = df[df$data == "tiny-few-speakers",]
df_many = df[df$data == "tiny-many-sessions",]
df_few = df[df$data == "tiny-few-sessions",]

# legend order
custom_legend_order <- c(
  "X-vector",
  "ECAPA-TDNN",
  "wav2vec 2.0"
)

# x_axis for every plot
breaks_5log10 <- function(x) {
  low <- floor(log10(min(x)/5))
  high <- ceiling(log10(max(x)/5))
  5 * 10^(seq.int(low, high))
}

breaks_log10 <- function(x) {
  low <- floor(log10(min(x)))
  high <- ceiling(log10(max(x)))
  
  10^(seq.int(low, high))
}

x_axis = scale_x_log10(
  name='learning rate', 
  breaks = breaks_log10,
  minor_breaks = breaks_5log10,
  labels = label_math(expr = 10^.x, format = log10)
)

# y_axis
y_axis =  scale_y_continuous(limits = c(0, 1), name="EER", labels = label_percent(accuracy=1))
y_lim = coord_cartesian(ylim=c(0, 0.25))

# the plots
vox2 = (
  ggplot(df_vox2)
  + aes(
    lr, 
    eer, 
    color=network,
    shape=network
  )
  + geom_point()
  + geom_line()
  + x_axis
  + y_axis + y_lim
  + scale_colour_colorblind(
    name='network',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='network',
    breaks=custom_legend_order,
    values=seq(0,6)
  )  
  + ggtitle('vox2')
  + theme_bw()
)

tiny_spk = (
  ggplot(df_spk)
  + aes(
    lr, 
    eer, 
    color=network,
    shape=network
  )
  + geom_point()
  + geom_line()
  + x_axis
  + y_axis + y_lim
  + scale_colour_colorblind(
    name='network',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='network',
    breaks=custom_legend_order,
    values=seq(0,6)
  ) 
  + ggtitle('few-speakers')
  + theme_bw()
)

tiny_few = (
  ggplot(df_few)
  + aes(
    lr, 
    eer, 
    color=network,
    shape=network
  )
  + geom_point()
  + geom_line()
  + x_axis
  + y_axis + y_lim
  + scale_colour_colorblind(
    name='network',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='network',
    breaks=custom_legend_order,
    values=seq(0,6)
  ) 
  + ggtitle('few-sessions')
  + theme_bw()
)

tiny_many = (
  ggplot(df_many)
  + aes(
    lr, 
    eer, 
    color=network,
    shape=network
  )
  + geom_point()
  + geom_line()
  + x_axis
  + y_axis + y_lim
  + scale_colour_colorblind(
    name='network',
    breaks=custom_legend_order,
    guide=guide_legend(nrow = 1)
  )
  + scale_shape_manual(
    name='network',
    breaks=custom_legend_order,
    values=seq(0,6)
  ) 
  + ggtitle('many-sessions')
  + theme_bw()
)

g = (
  vox2 / tiny_spk /
  tiny_few / tiny_many
  + plot_layout(guides = "collect")
  & theme(
    legend.direction = "horizontal",
    legend.position = "bottom",
    plot.margin = unit(c(0.1,0.5,0.1,0.1), "cm"),
    text         = element_text(family="mono"),
    axis.title.x = element_text(family="CM Roman"),
    axis.title.y = element_text(family= "CM Roman")
  )
)

g

ggsave(
  file="plot_four_thesis.pdf",
  #device=cairo_pdf,
  width = 130,
  height = 180,
  units = "mm"
)
embed_fonts("plot_four_thesis.pdf", outfile="plot_four_thesis.pdf")
