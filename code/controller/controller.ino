#include <Ultrasonic.h>




//################# Variables #################
String readString;

Ultrasonic ultrasonic(A0,A1);


//################# Setup #################
void setup()
  {
    Serial.begin(9600);
  }


//################# Main Loop #################
void loop() 
  {
   // Loop while there no data is being transmitted
   while(!Serial.available())
     {
     }
   // Loop while data is being transmitted
   while(Serial.available())
    {
    delay(30); 
    if (Serial.available()>0)
      {
        char c = Serial.read();
        //store received data in readString
        readString += c;  
      }
    }
  // sending back data readstring is no longer empty
  if (readString.length()>0)
    { 
      if (readString = "height")
      {
        //this sends ultrasonic distance and weight value 
        Serial.println(ultrasonic.Ranging(CM));
        
      }
      
      if (readString = "weight")
      {
        Serial.println("365");
      }
    }
    readString = ""; 
  }
  


