grammar Sample;

sample         : MAIN city EOF;

main           : 'fn' 'main' scope;

city           : 'city' IDENTIFIER '{' cityElements '}';

cityElements   : mapElement (',' mapElement)* | mapElement | /* ε */ ;

mapElement     : buildingTypes IDENTIFIER '{' buildingElements '}';

buildingElements: command (',' command)* | command | /* ε */ ;

buildingTypes  : 'road' | 'building';

scope          : '{' statements '}';

statements     : statement statements | statement | /* ε */ ;

statement      : expression ';' | forLoop | function | loop | assignment ';' | command ';';

command        : line | bend | box | circ;

line           : 'line' '(' POINT ',' POINT ')';

bend           : 'bend' '(' POINT ',' POINT ',' ANGLE ')';

box            : 'box' '(' POINT ',' POINT ')';

circ           : 'circ' '(' POINT ',' (FLOAT | INT) ')';

string         : '"' chars '"';

chars          : CHARS CHAR | CHAR | /* ε */ ;

range          : expression '..' expression;

TYPE           : 'bool'
              | 'u8' | 'u16' | 'u32' | 'u64' | 'u128'
              | 'i8' | 'i16' | 'i32' | 'i64' | 'i128'
              | 'f32' | 'f64'
              | 'char'
              | 'string'
              | ARRAY_TYPE
              | IDENTIFIER
              ;

ARRAY_TYPE     : TYPE '[]';

array          : '[' arrayElements ']';

arrayElements  : expression ',' arrayElements | expression | /* ε */ ;

forLoop        : 'for' IDENTIFIER 'in' expression scope;

loop           : 'loop' scope | 'loop' RANGE scope;

ANGLE          : (FLOAT '°' | INT '°');

function       : functionHead functionBody;

functionHead   : 'fn' IDENTIFIER '(' parameters ')' ':' TYPE
               | 'fn' IDENTIFIER '(' parameters ')'
               ;

parameters     : parameter ',' parameters | parameter | /* ε */ ;

parameter      : IDENTIFIER ':' TYPE;

functionBody   : '{' maybeReturns '}';

maybeReturns   : statements 'return' | 'return' | statements;

FLOAT          : INT '.' INT;

CHAR           : '^.$';

assignment     : ('let' | 'const') varDeclaration;

varDeclaration : IDENTIFIER ':' TYPE '=' expression | IDENTIFIER '=' expression;

POINT          : '(' POINT_COMPONENT ',' POINT_COMPONENT ')';

POINT_COMPONENT: FLOAT | INT;

ifExpr         : 'if' expression scope | 'if' expression scope elseIf;

elseScope      : 'else' scope;

elseIf         : 'else' ifExpr elseIf | 'else' ifExpr | /* ε */ ;

IDENTIFIER     : LETTER CHARACTERS;

expression     : expression '+' expressionMultiplication
                 | expression '-' expressionMultiplication
                 | expressionTerm
                 ;

expressionTerm : expressionTerm '*' power
                 | expressionTerm '/' power
                 | power
                 ;

power          : factor '^' power
                 | factor
                 ;

factor         : '(' expression ')'
                 | assignment
                 | validExpression
                 ;

validExpression: ifExpr
                 | INT
                 | FLOAT
                 | assignment
                 ;

LETTER         : [A-Za-z];
CHARACTERS     : LETTER+;
INT            : [0-9]+;
DIGIT          : '0' .. '9';
WS             : [ \t\n\r]+ -> skip;
