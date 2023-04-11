# R 
# 10/25/2022
# This script will produce the double scatter plot for supplemental image of pipeline evolution in terms of R2 and DS for the XOR dataset.

# Feature importance supplemental plot for XOR experiment
library(ggplot2)

setwd("~/Library/CloudStorage/Box-Box/CedersSinai/autoQTL/Manuscripts/BioData Mining/Supplemental/Plot Files")
XORFI <- read.csv("~/Library/CloudStorage/Box-Box/CedersSinai/autoQTL/Manuscripts/BioData Mining/Supplemental/Plot Files/XORFI.csv")

Supp_FI <- ggplot(XORFI, aes(SNP, Shap, fill = group)) +
  geom_violin(color="black", size=3, alpha = 0.5) +
  geom_boxplot(width=0.1) +
  theme_light() + theme(panel.grid=element_blank()) +
  labs(x = "SNP\n", y = "Avg. Shapley Value") +
  theme(axis.title.x = element_text(color="black", face="bold", size=30, margin=margin(10,0,0,0))) +
  theme(axis.title.y = element_text(color="black", face="bold", size=30, margin=margin(0,10,0,0))) +
  theme(axis.text.x = element_text(size=30, color="black", face="bold", angle = 90, vjust = 0.5, hjust = 1)) +
  theme(axis.text.y = element_text(size=30, color="black", face="bold")) +
  theme(panel.border = element_rect(color = "black", size=1.5)) +
  theme(axis.ticks = element_line(color = "black", size=1.5)) +
  theme(legend.position = "none") +
  #scale_fill_manual(breaks = XORFI$group, values = c("green", "blue", "red", "purple", "orange", "cyan", "brown1", "darkgreen", "deeppink")) +
  #scale_fill_manual(values=c("green", "green", "blue", "blue", "red", "red", "purple", "purple", "orange", "orange", "cyan", "cyan", "brown1", "brown1", "darkgreen", "darkgreen", "deeppink", "deeppink")) +
  scale_y_continuous(limits=c(0, 0.16), breaks=c(0, 0.05, 0.10, 0.15))
  #labs(tag = "B") +
  #theme(plot.tag.position = c(0.021, 0.98), plot.tag = element_text(size = 30, face = "bold"))

Supp_FI

ggsave("Supp_FI.pdf", plot = last_plot(), device = "pdf", width = 15, height = 10, units = "in", useDingbats=FALSE)

# Import data for Panel 2 (multi-scatter plot)
setwd("~/Library/CloudStorage/Box-Box/CedersSinai/AutoQTL/Manuscripts/BioData Mining/Supplemental/Plot Files")
MSP1 <- read.csv("~/Library/CloudStorage/Box-Box/CedersSinai/AutoQTL/Manuscripts/BioData Mining/Supplemental/Plot Files/SuppMSP_R2.csv")
MSP2 <- read.csv("~/Library/CloudStorage/Box-Box/CedersSinai/AutoQTL/Manuscripts/BioData Mining/Supplemental/Plot Files/SuppMSP_DS.csv")

MSP1$z <- "data1"
MSP1$z <- factor(MSP1$z)
MSP2$z <- "data2"
MSP2$z <- factor(MSP2$z)

MSP3 <- within(MSP2, { y = y/100 })
MSP4 <- rbind(MSP1, MSP3)

mycolors <- c("data1"="blue", "data2"="red")

Supp_MSP <- ggplot(MSP4, aes(x = x, y = y, group = z, color = z)) +
  stat_summary(aes(x = x, y = y, group = z, color = z), fun = mean, geom = "line", size = 2.5) +
  stat_summary(aes(x = x, y = y, group = z, fill = z), fun.data = mean_se, geom = "ribbon", alpha = 0.2, color = NA) +
  stat_summary(aes(x = x, y = y, group = z), fun = mean, geom = "point", color = "black", size = 4, stroke = 1.5) +
  theme_light() + theme(panel.grid=element_blank()) +
  theme(legend.position="none") +
  labs(x ="Generation") +
  scale_color_manual(name="z", values = mycolors, guide = FALSE) +
  scale_fill_manual(name = "z", values = mycolors, guide = FALSE) +
  theme(axis.title.x = element_text(color="black", face="bold", size=30, margin=margin(10,0,0,0))) +
  theme(
    axis.title.y = element_text(color = mycolors["data1"], face = "bold", size = 30, margin=margin(0,10,0,0)),
    axis.text.y = element_text(color = mycolors["data1"]),
    axis.title.y.right = element_text(color = mycolors["data2"], face = "bold", size = 30, margin=margin(0,0,0,10)),
    axis.text.y.right = element_text(color = mycolors["data2"])
  ) +
  theme(axis.text = element_text(size=30, color="black", face="bold")) +
  theme(panel.border = element_rect(color = "black", size=1.5)) +
  theme(axis.ticks = element_line(color = "black", size=1.5)) +
  scale_x_continuous(limits=c(1, 25), breaks=c(1, 5, 10, 15, 20, 25)) +
  scale_y_continuous(limits=c(0.019, 0.085), breaks=c(0.02, 0.04, 0.06, 0.08), name = "Mean R2", sec.axis = sec_axis(~ 100*., name = "Mean Difference Score"))
  #labs(tag = "B") +
  #theme(plot.tag.position = c(0.025, 0.98), plot.tag = element_text(size = 30, face = "bold"))

Supp_MSP

ggsave("Supp_MSP.pdf", plot = last_plot(), device = "pdf", width = 10, height = 10, units = "in", useDingbats=FALSE)