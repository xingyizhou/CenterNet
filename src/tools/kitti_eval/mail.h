#ifndef MAIL_H
#define MAIL_H

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

class Mail {

public:

  Mail (std::string email = "") {
    if (email.compare("")) {
      mail = popen("/usr/lib/sendmail -t -f noreply@cvlibs.net","w");
      fprintf(mail,"To: %s\n", email.c_str());
      fprintf(mail,"From: noreply@cvlibs.net\n");
      fprintf(mail,"Subject: KITTI Evaluation Benchmark\n");
      fprintf(mail,"\n\n");
    } else {
      mail = 0;
    }
  }
  
  ~Mail() {
    if (mail) {
      pclose(mail);
    }
  }
  
  void msg (const char *format, ...) {
    va_list args;
    va_start(args,format);
    if (mail) {
      vfprintf(mail,format,args);
      fprintf(mail,"\n");
    }
    vprintf(format,args);
    printf("\n");
    va_end(args);
  }
    
private:

  FILE *mail;
  
};

#endif
