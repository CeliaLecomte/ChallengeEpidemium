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

//Main Function
function SitesRetriever(input) {
  try {
  var url = input.url;
  var ref = input.ref;  
  //Define variable to return results and keep ref of the study
  var Centercoord = [];
  
  
  
  
  //Retrieve html page
  var html = UrlFetchApp.fetch(url).getContentText();
  
  //remove all url to discard character & messing up with xml parse and to close <img> <link> <meta> <input> tag and <> in "data-text"
  var regex1 = new RegExp('src\s*=\s*"([^"]+)"', 'gi');
  var regex2 = new RegExp('(<img[^>]*?[^\/]\s*)(>)', 'gi');
  var regex3 = new RegExp('(<link[^>]*?[^\/]\s*)(>)', 'gi');
  var regex4 = new RegExp('(<meta[^>]*?[^\/]\s*)(>)', 'gi');
  var regex5 = new RegExp('data-text\s*=\s*"([^"]+)"', 'gi');
  var regex6 = new RegExp('data-alt-text\s*=\s*"([^"]+)"', 'gi');
  var regex7 = new RegExp('(<input[^>]*?[^\/]\s*)(>)', 'gi');
  var regex8 = new RegExp('\&times;', 'gi');
  var regex9 = new RegExp(';>', 'gi');
  

  html = html.replace(regex1, 'src="WGBDiP"');
  html = html.replace(regex2, '$1/$2');
  html = html.replace(regex3, '$1/$2');
  html = html.replace(regex4, '$1/$2');
  html = html.replace(regex5, 'data-text="WGBDiP"');
  html = html.replace(regex6, 'data-alt-text="WGBDiP"');
  html = html.replace(regex7, '$1/$2');
  html = html.replace(regex8, '"WGBDiP"');
  html = html.replace(regex9, '>');

  
  
  //Parse html web page information as xml
  
  var doc = Xml.parse(html, true);
  var bodyHtml = doc.html.body.toXmlString();
  doc = XmlService.parse(bodyHtml);
  var docR = doc.getRootElement();
  var maps = getElementsByClassName(docR, 'maps row')[0];
  
  
  //Then extract the section containing all the sites from the web page
  var output = XmlService.getRawFormat().format(maps);
  var Centers = getElementsByClassName(maps, 'modal');
   
  
  // Then go through all the sites and retrieved and get relevant information nom du site, latitude, longitude, nom de la ville,
 
  for(k = 0; k<Centers.length; k++) {
    var S = (k*4)+1;
    var Center = XmlService.parse(XmlService.getRawFormat().format(Centers[k]));           
    var CenterR = Center.getRootElement();
    var CenterName = getElementsByClassName(CenterR, 'modal-title title1 colored')[0].getValue();
    var City = getElementsByClassName(CenterR, 'map-search-result-title')[0].getValue();
    var lat = CenterR.getAttribute('data-lat').getValue();
    var lng = CenterR.getAttribute('data-lng').getValue();
    //Save values in the array
     Centercoord[k] = [];
    Centercoord[k][0] = ref ;
    Centercoord[k][1] = CenterName;
    Centercoord[k][2] = City;
    Centercoord[k][3] = lat;
    Centercoord[k][4] = lng;
    Centercoord[k][5] = k + 1;
  } 
  
  
  
  
  return (Centercoord);
  }
  catch (e){return([[ref,"no site","no site","no site","no site",0]])}
    
    
    
}