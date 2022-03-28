settings {
   logfile    = "/tmp/lsyncd.log",
   statusFile = "/tmp/lsyncd.status",
   nodaemon   = true
}

sync {
   default.rsyncssh,
   source       ="/home/nik/workspace/phd/repo/w2v2-speaker-few-samples",
   host         ="nvaessen@cn99.science.ru.nl",
   excludeFrom  =".gitignore",
   targetdir    ="/home/nvaessen/remote/repo/w2v2-speaker-few-samples",
   delay        = 0,
   rsync = {
     archive    = true,
     compress   = false,
     whole_file = false
   }
}