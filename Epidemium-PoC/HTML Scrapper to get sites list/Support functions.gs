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

//This function get xml document by filtering on the css class

function getElementsByClassName(element, classToFind) {  
  var data = [];
  var descendants = element.getDescendants();
  descendants.push(element);  
  for(i in descendants) {
    var elt = descendants[i].asElement();
    if(elt != null) {
      var classes = elt.getAttribute('class');
      if(classes != null) {
        classes = classes.getValue();
        if(classes == classToFind) data.push(elt);
        else {
          classes = classes.split(' ');
          for(j in classes) {
            if(classes[j] == classToFind) {
              data.push(elt);
              break;
            }
          }
        }
      }
    }
  }
  return data;
}


/**
* This function sets the new execution time as the 'runtimeCount' script property.
*/
	
function runtimeCountStop() {

  var props = PropertiesService.getScriptProperties();
  var currentRuntime = props.getProperty("runtimeCount");
  var stop = new Date();
  var start = new Date(props.getProperty("CountStart"));
  var newRuntime = ((Number(stop) - Number(start))/1000) + currentRuntime;
  var setRuntime = {
    runtimeCount: newRuntime,
  }
  props.setProperties(setRuntime);

}


function myf() {
  var range = SpreadsheetApp.getActiveSheet().getRange("F2:F5");
  
  var data = [];
  
  data[0] = [1,2,3,4,5];
  data[1] = ["A",2,3,4,5];
  data[2] = ["B",4,5,1,2];
  data[3] = ["c",2,3,4,5];
  
  
  var dataD = [];
  
  dataD[0] = [1234,2,3,4,5];
  dataD[1] = ["D",2,3,4,5];
  dataD[2] = ["D",4,5,1,2];
  dataD[3] = ["D",2,3,4,5];
  
//  var array = data;
//   // get length of the longest row
// var max = array
//  .slice(0)
//  .sort(function (a,b) {
//    return a.length < b.length ? 1 : a.length > b.length ? -1 : 0;
//  })[0].length;
//
// // arrays are zero indexed
// var maxi = max-1;
//
// // insert a pointer at max index if none exists
// array = array
//  .map(function (a){ 
//    a[maxi] = a[maxi] || ""; 
//    return a; 
//  });

  var ini;
  var ini2 =[[]];
  ini2 = ini2.concat(data);
  ini = data.concat(dataD);
  
 Logger.log(ini2);
  
 // range.setValues(data);
  
  


}






