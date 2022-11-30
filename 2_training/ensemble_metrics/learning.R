#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(pilot)
##################################################################################
# Learning curve
##################################################################################
learningcurve <- function(x){
	ler <- read_csv(x) %>%
		select(train_size, models, starts_with("train_scores"), starts_with("test_scores")) %>%
		rowwise() %>%
		transmute(
			train_size = train_size,
			models = models,
			train_score = mean(c_across(starts_with("train_scores"))),
			train_deviation = sd(c_across(starts_with("train_scores"))),
			test_score = mean(c_across(starts_with("test_scores"))),
			test_deviation = sd(c_across(starts_with("test_scores")))) %>%
		ggplot(aes(x=train_size))+
				geom_ribbon(
					aes(
						ymin = train_score-train_deviation,
						ymax = train_score+train_deviation),
					colour = "grey90")+
		geom_line(aes(y = train_score, colour=models))+
				geom_ribbon(
					aes(
						ymin = test_score-test_deviation,
						ymax = test_score+test_deviation),
					fill = "grey90")+
		geom_line(aes(y = test_score, colour=models))+
		facet_wrap(~models)+
		scale_colour_manual(
		  "Modelo",
		   values = c("#5E1E5B","#F89B0F","#F26F7E"))+
		theme(
		  # Use gray text for the region names
		  axis.text.x = element_text(color = "gray12", size = 12),
		  axis.text.y = element_text(color = "gray12", size = 12),
		  # Move the legend to the bottom
		  legend.position = "bottom",
		  # Set default color and font family for the text
		  text = element_text(color = "gray12"),
		  # Make the background white and remove extra grid lines
		  panel.background = element_rect(fill = "white", color = "white"),
		  panel.grid = element_blank(),
		  panel.grid.major.x = element_blank()
		)+
		annotate("label", x=200, y=1, label="Train")+
		annotate("label", x=200, y=0.87, label="Test")+
		theme_pilot()
		return(ler)
}
##################################################################################
# Scalability of the model
##################################################################################
scalability <- function(x){
	sca <- read_csv(x) %>%
		select(train_size, models, starts_with("fit_times_fold")) %>%
		rowwise() %>%
		transmute(
			train_size = train_size,
			models = models,
			fit_times = mean(c_across(starts_with("fit_times_fold"))),
			deviation = sd(c_across(starts_with("fit_times_fold")))) %>%
		ggplot(aes(x=train_size, group=models))+
				geom_ribbon(
					aes(
						ymin = fit_times-deviation, 
						ymax = fit_times+deviation),
					fill = "grey90")+
		geom_line(aes(y = fit_times, colour=models))+
	  facet_wrap(~models, scales="free")+
		scale_colour_manual(
		  "Modelo",
		   values = c("#5E1E5B","#F89B0F","#F26F7E"))+
		theme(
		  # Use gray text for the region names
		  axis.text.x = element_text(color = "gray12", size = 12),
		  axis.text.y = element_text(color = "gray12", size = 12),
		  # Move the legend to the bottom
		  legend.position = "bottom",
		  # Set default color and font family for the text
		  text = element_text(color = "gray12"),
		  # Make the background white and remove extra grid lines
		  panel.background = element_rect(fill = "white", color = "white"),
		  panel.grid = element_blank(),
		  panel.grid.major.x = element_blank()
		)+
		theme_pilot()
	return(sca)
}
##################################################################################
# Performance of the model
##################################################################################

performance <- function(x){
	perf <- read_csv(x) %>%
		select(models, starts_with("fit_times_fold") , starts_with("test_score")) %>%
			pivot_longer(cols=starts_with("fit_times_fold"), names_to="fold", values_to="fit_times") %>%
			rowwise() %>%
			transmute(
				models = models,
				fit_times = fit_times,
				test_score = mean(c_across(starts_with("test_score"))),
				deviation = sd(c_across(starts_with("test_score")))) %>%
			ggplot(aes(x=fit_times, group=models))+
				geom_ribbon(
					aes(
						ymin = test_score-deviation,
						ymax = test_score+deviation),
					fill = "grey90")+
#				geom_point()+
				geom_line(aes(y = test_score, colour=models))+
			facet_wrap(~models, scales="free_x")+
		scale_colour_manual(
		  "Modelo",
		   values = c("#5E1E5B","#F89B0F","#F26F7E"))+
		theme(
		  # Use gray text for the region names
		  axis.text.x = element_text(color = "gray12", size = 12),
		  axis.text.y = element_text(color = "gray12", size = 12),
		  # Move the legend to the bottom
		  legend.position = "bottom",
		  # Set default color and font family for the text
		  text = element_text(color = "gray12"),
		  # Make the background white and remove extra grid lines
		  panel.background = element_rect(fill = "white", color = "white"),
		  panel.grid = element_blank(),
		  panel.grid.major.x = element_blank()
		)+
		theme_pilot()
	return(perf)
}
# Save all figures in pdf
lsp <- function(x, n){
	ggsave(n[1],learningcurve(x),dpi=300,width = 4000, height = 2000, units = "px",useDingbats=FALSE)
	ggsave(n[2],scalability(x),dpi=300,width = 4000, height = 2000, units = "px",useDingbats=FALSE)
	ggsave(n[3],performance(x),dpi=300,width = 4000, height = 2000, units = "px",useDingbats=FALSE)
}
x = c("bagging_learning_curve.csv",
"boosting_learning_curve.csv",
"learning_curve.csv",
"voting_learning_curve.csv",
"stacking_learning_curve.csv")
names = c("bagging_lc.pdf",
"bagging_sc.pdf",
"bagging_pr.pdf",
"boosting_lc.pdf",
"boosting_sc.pdf",
"boosting_pr.pdf",
"opt_lc.pdf",
"opt_sc.pdf",
"opt_pr.pdf",
"voting_lc.pdf",
"voting_sc.pdf",
"voting_pr.pdf",
"stacking_lc.pdf",
"stacking_sc.pdf",
"stacking_pr.pdf")
# Generate images
for(i in 1:5) lsp(x[i],names[(3*i-2):(3*i)])
