--ID aspect text
CREATE TABLE Ai_embeddings(
        ID integer primary key AUTOINCREMENT,
        word varchar(23),
        vector varchar(1023)--context
        );