This is to learn git pushing after creating a repository.

1. First copy the url of the repository https://github.com/AINinja2049/DeepLearning_FromScratch_Part_1.git
2. git init
3. git add .
4. git pull origin main --rebase %%Need to pull the remote branch first and merge it with your local work%
5. If you already have another url connected elsewhere then set it to this url.
6. git remote set-url origin https://github.com/AINinja2049/DeepLearning_FromScratch_Part_1.git
7. Otherwise use the following
8. git remote add origin https://github.com/AINinja2049/DeepLearning_FromScratch_Part_1.git
9. git push -u origin main #The main command have now started using but people used master before#
10. git status #Command to view the status#
11. git remote -v #This views which of the repository is connected#
