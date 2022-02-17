#include <stdio.h>
#include <stdlib.h>
#include "sqlite3.h" 
#include <fstream>

// sqlite commands implemented for TRA API
extern "C" {
   static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
      int i;
      for(i = 0; i<argc; i++) {
         printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
      }
      printf("\n");
      return 0;
   }

   // create a db
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

   // execuate sqlite3 commands
   void execute_command(const char* sqlite_command, const char* db_name, const char* data){
      sqlite3 *db;
      char *zErrMsg = 0;
      int rc;
      char *sql;
      sqlite3_stmt *res;

      /* Open database */
      rc = sqlite3_open(db_name, &db);
      
      if( rc ) {
         fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
         return;
      } else {
         fprintf(stdout, "Opened database successfully\n");
      }

      /* Execute SQL statement */
      rc = sqlite3_exec(db, sqlite_command, callback, (void*)data, &zErrMsg);
      
      if( rc != SQLITE_OK ){
         fprintf(stderr, "SQL error: %s\n", zErrMsg);
         sqlite3_free(zErrMsg);
      } else {
         fprintf(stdout, "SQLite command is successfully executed\n");
      }
      sqlite3_close(db);
   }

     
}

// int main(int argc, char* argv[]){
//       create_db("test.db");
//       char* sql = "CREATE TABLE COMPANY("  \
//       "ID INT PRIMARY KEY     NOT NULL," \
//       "NAME           TEXT    NOT NULL," \
//       "AGE            INT     NOT NULL," \
//       "ADDRESS        CHAR(50)," \
//       "SALARY         REAL );";
//       execute_command(sql, "test.db");

//       sql = "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "  \
//          "VALUES (1, 'Paul', 32, 'California', 20000.00 ); " \
//          "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "  \
//          "VALUES (2, 'Allen', 25, 'Texas', 15000.00 ); "     \
//          "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)" \
//          "VALUES (3, 'Teddy', 23, 'Norway', 20000.00 );" \
//          "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)" \
//          "VALUES (4, 'Mark', 25, 'Rich-Mond ', 65000.00 );";
//       execute_command(sql, "test.db");
// } 





