#!/usr/bin/env Rscript
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(pilot)
library(patchwork)
#############
# Set general theme
#############
theme_set(
	theme(
		  # Use gray text for the region names
		  axis.text.x = element_text(color = "gray12", size = 12),
		  axis.text.y = element_text(color = "gray12", size = 12),
		  # Set default color and font family for the text
		  text = element_text(color = "gray12"),
		  # Make the background white and remove extra grid lines
		  panel.background = element_rect(fill = "white", color = "white"),
		  panel.grid = element_blank(),
		  panel.grid.major.x = element_blank(),
		  strip.background = element_blank()
#		  strip.text.x = element_blank()
		  )+
  	theme_pilot()
)
##################################################################################
# Learning curve
##################################################################################
learningcurve <- function(x){
	ler <- read_csv(x) %>%
		select(train_size, models, starts_with("train_scores"), starts_with("validation_scores")) %>%
		rowwise() %>%
		transmute(
			train_size = train_size,
			models = models,
			train_score = mean(c_across(starts_with("train_scores"))),
			train_deviation = sd(c_across(starts_with("train_scores"))),
			validation_score = mean(c_across(starts_with("validation_scores"))),
			validation_deviation = sd(c_across(starts_with("validation_scores")))) %>%
		ggplot(aes(x=train_size))+
				geom_ribbon(
					aes(
						ymin = train_score-train_deviation,
						ymax = train_score+train_deviation),
					colour = "grey90")+
		geom_line(aes(y = train_score, colour=models))+
				geom_ribbon(
					aes(
						ymin = validation_score-validation_deviation,
						ymax = validation_score+validation_deviation),
					fill = "grey90")+
		geom_line(aes(y = validation_score, colour=models))+
		facet_wrap(~models, scales = "free_y")+
		scale_color_pilot()+
		labs(title="Learning Curve")
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
	  facet_wrap(~models, scales="free_y")+
		scale_color_pilot()+
		labs(title="Scalability Plot")
	return(sca)
}
##################################################################################
# Performance of the model
##################################################################################
performance <- function(x){
	perf <- read_csv(x) %>%
		select(models, starts_with("fit_times_fold") , starts_with("validation_score")) %>%
			pivot_longer(cols=starts_with("fit_times_fold"), names_to="fold", values_to="fit_times") %>%
			rowwise() %>%
			transmute(
				models = models,
				fit_times = fit_times,
				validation_score = mean(c_across(starts_with("validation_score"))),
				deviation = sd(c_across(starts_with("validation_score")))) %>%
			ggplot(aes(x=fit_times, group=models))+
				geom_ribbon(
					aes(
						ymin = validation_score-deviation,
						ymax = validation_score+deviation),
					fill = "grey90")+
				geom_line(aes(y = validation_score, colour=models))+
			facet_wrap(~models, scales="free")+
		scale_color_pilot()+
		labs(title="Performance Plot")
	return(perf)
}
# Save all figures in pdf
x = "./minilearning_curve.csv.gz"
fig <- (learningcurve(x)+scalability(x)+performance(x))+plot_layout(guides = 'keep')
# Generate images
ggsave(
	"curves.pdf",
	fig,
	dpi=320,
	width = 12000, 
	height = 3500, 
	units = "px",
	useDingbats=FALSE)
ggsave("curves.png",fig,dpi=320,width = 12000, height = 3500,bg = "white", units = "px")
