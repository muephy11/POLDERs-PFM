#ARDF to txt converter 
#Place the Place the script in the folder you want to convert the data 
#Important!: Set WD manually before executing any code of the script!

library(Rasylum)

files = read.csv("files.csv")
files = files$x
for (file in files) 
{
  try({
    fl = unlist(strsplit(file,"[.]"))
    fl = fl[1]
    print(fl)
    #Reads the ARDF file in; long procedure!
    x <- read.ardf(file)
    
    #Create a folder with the filename and save the ForceCurves in it.
    i=1
    curvenames = names(x$data)
    len = length(curvenames)
    dir.create(fl)
    while(i<=len)
    {
      FName_CSV = paste(fl,"\\",fl,"_",curvenames[i],".csv", sep = "")
      write.csv(x$data[i],file = FName_CSV)
      i=i+1
    }
  })
}