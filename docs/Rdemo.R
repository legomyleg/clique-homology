#install.packages("tidyverse") #uncomment and run this line to install the package
library(tidyverse) #import the package

# initialize the data frame
df <- data.frame("x" = seq(1, 10), "y" = seq(1, 10))

ggplot(df, aes(x = x, y = y)) + 
geom_point(size = 2) +
labs(
    xlab = "X",
    ylab = "Y",
    title = "Y = X"
) + 
theme_minimal()
