//#################################################################################################
//#                    A-R-O-U-N-D trial project
//#                     An EPIDEMIUM project
//#                        Challenge 1-2
//#          
//#    Written : Elie Toledano
//#    Date of retrieval: January 2018
//#   
//# 
//# Scope : INCa Clinical trial registry
//#################################################################################################


// This function will ensure that there is no issue with over quotas and split the job accordingly

function BatchTreatment() {
  // Start counting execution time for debugging if over quotas error
  var runtimeCountStart = new Date();
  PropertiesService.getScriptProperties().setProperty('CountStart', runtimeCountStart);
  
  //Define batch according to expected run time
  var BATCH_SIZE = 30;
  
  //Get Number of line to be analyzed, do not forget to remove
  var StudyNumber = SpreadsheetApp.openById('1yST0RsGPRGOvlX7dSttCFKxHpfXLAymZZZqLbV_TRK0').getActiveSheet().getDataRange().getLastRow()-1; 
  
  // Set Properties for later use
  var props = PropertiesService.getScriptProperties();
  props.setProperty('StudyNumber', StudyNumber);
  props.setProperty('Batch', BATCH_SIZE);
  props.setProperty('LastTreated', 1);
  
  
  //Set triggers to run the script
  
  ScriptApp.newTrigger("CoordRun").timeBased().everyMinutes(5).create();
}



function CoordRun() {
  //Get details
  var props = PropertiesService.getScriptProperties();
  var BATCH_SIZE = Number(props.getProperty('Batch'));
  var StudyNumber = Number(props.getProperty('StudyNumber'));
  var LastTreated = Number(props.getProperty('LastTreated'));
  var spreadsheet = SpreadsheetApp.openById('1yST0RsGPRGOvlX7dSttCFKxHpfXLAymZZZqLbV_TRK0');
  
  //First check that we are not at the last remaining batch
  
  if ((LastTreated + BATCH_SIZE)>=(StudyNumber+1)){
    var rowS = LastTreated + 1;
    var Nrow = (StudyNumber + 1) - LastTreated;
    var stop = 'TRUE';
  }
  
  else{
    var rowS = LastTreated + 1;
    var Nrow = BATCH_SIZE;
    var stop = 'FALSE';
  }
  
  //Get studies to be treated by the script
  var Studies = spreadsheet.getSheetByName('Sheet1').getRange(rowS, 1, Nrow, 3).getValues();
  
  //Set data retrieval using parsing
  var data = [[]];
  var a = 0;
 
  for (l = 0; l < Studies.length; l++) {
    var study = Studies[l];
    var input = {ref: study[0], url: study[2]};
   
    var parse = SitesRetriever(input);
    data = data.concat(parse);
  }
  //copy data to sheet
  //we have to go square the matrix
  
//    
//     var max = data
//                .slice(0)
//                .sort(function (a,b) {
//                  return a.length < b.length ? 1 : a.length > b.length ? -1 : 0;})[0].length;
//  // arrays are zero indexed
//  var maxi = max-1;
//  
//  // insert a pointer at max index if none exists
//  data = data
//  .map(function (a){ 
//    a[maxi] = a[maxi] || ""; 
//    return a; 
//  });
  data = data.slice(1);
  var lastrow = spreadsheet.getSheetByName('Sheet2').getDataRange().getNumRows() + 1;
  var range = spreadsheet.getSheetByName('Sheet2').getRange(lastrow, 1, data.length, data[0].length);
  
  range.setValues(data);
  
  //Update properties for next run
  
  props.setProperty('LastTreated', LastTreated + BATCH_SIZE);
  
 

       
  
  if(stop == 'TRUE') {
    var Trigger = ScriptApp.getProjectTriggers()[0];
    ScriptApp.deleteTrigger(Trigger);
    // Stop counting execution time
    runtimeCountStop();
  }
    
}