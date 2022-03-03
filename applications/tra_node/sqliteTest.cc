#include <stdio.h>
#include "sqlite3.h" 

extern "C" {
   void create_db(const char* db_name){
      sqlite3 *db;
      char *zErrMsg = 0;
      int rc;

      rc = sqlite3_open(db_name, &db);

      if( rc ) {
         fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
      } else {
         fprintf(stderr, "Opened database successfully\n");
      }
      sqlite3_close(db);
   }
    int main(int argc, char* argv[]) {
      create_db("test.db");
   }
}
