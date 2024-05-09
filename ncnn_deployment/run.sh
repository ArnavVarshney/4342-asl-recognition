program=$1

if [ -z "$program" ]
then
      #Default is test.cpp
      program="test"
fi

g++ -o "$program" "$program.cpp"
./$program