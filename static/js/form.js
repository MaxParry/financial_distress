var $ageInput = document.querySelector('#inputAge')
var $salaryInput = document.querySelector('#inputSalary');
var $dependentsInput = document.querySelector('#inputDependents');
var $creditLineInput = document.querySelector('#inputOpenCredit');
var $creditLimitInput = document.querySelector('#inputCreditLimit');
var $realEstateInput = document.querySelector('#inputOpenRE');
var $monthlySpendInput = document.querySelector('#inputMonthySpend');
var $totalDebtInput = document.querySelector('#inputDebt');
var $overdueInput = document.querySelector('#inputOverdue');
var $submitBtn = document.querySelector('#submit');

var inputFields = [$ageInput,$salaryInput,$dependentsInput,$creditLineInput,$creditLimitInput,$realEstateInput,$monthlySpendInput,$totalDebtInput,$overdueInput];

function renderDropdown(arr, $parent, type='default') {
    if (type == 'choose') {
        $parent.innerHTML = '<option value="null" selected disabled>Choose...</option>';
    };

    if (arr[0].length>1){
        for (i=0 ; i < arr.length ; i++) {
            var $option = document.createElement('option');
            $option.setAttribute('value',arr[i][0])
            if (i == (arr.length - 1)){
                $option.innerText = arr[i][1]+'+';
            } else{
                $option.innerText = arr[i][1];
            };
            $parent.appendChild($option);
        };
    } else {
        for (i=0 ; i < arr.length ; i++) {
            var $option = document.createElement('option');
            $option.setAttribute('value',arr[i])
            if (i == (arr.length - 1)){
                $option.innerText = arr[i]+'+';
            } else{
                $option.innerText = arr[i];
            };
            $parent.appendChild($option);
        };
    };
};

function createConsecutiveArr(start,end){
    var arr = [];

    for (i=start; i < (end+1); i++) {
        arr.push(i);
    };

    return arr;
};

function createSteppedArr(instructions,start){
    var arr = [];
    var limit = 0;

    if (start!=1){
        arr.push(0);
    };

    for (i=0,n=0; i<instructions.length; i++){
        for (n; n < instructions[i][0]; n) {
            n = n + instructions[i][1]
            arr.push(n);
        };
    };

    return arr;
}

// Create money dropdowns
function showMeTheMoney(numArr){
    var arr = [];

    for (i=0; i< numArr.length ; i++) {
        var n = numArr[i].toLocaleString();
        n = '$' + n;
        arr.push(n);
    };

    return arr;
};

// Zip money array with value array
function zipIt(numArr,monArr){
    return numArr.map(function(e, i) {
      return [e, monArr[i]];
      });
};

// insert money options
function goMoney(instruct,$div) {
    var arr = createSteppedArr(instruct,1);
    var styledArr = showMeTheMoney(arr);
    var zippedArr = zipIt(arr,styledArr);
    renderDropdown(zippedArr,$div,'choose');
};

// Check if all fields are filled
function fieldsNull() {
    var nulls = [];
    //add null fields to array
    for (var i=0; i<inputFields.length; i++) {
        if (inputFields[i].value == 'null') {
            nulls.push(inputFields[i]);
        };
    };
    if (nulls.length>0){
        return nulls;
    } else {
        return false;
    };
};

// add styling
function addIncompleteFieldStyle(nullFields){
    for (var i=0; i<nullFields.length; i++){
        var elem = nullFields[i];
        var par = elem.parentElement;
        par.setAttribute('style', 'border: 1px solid #B82601; border-radius: 5px;')
    };

    resetPar = document.getElementById('submit').parentElement;
    console.log(resetPar);


    var warningDiv = document.createElement('div')
    warningDiv.setAttribute('id','warningDiv');
    warningDiv.setAttribute('class','row');

    var warning = document.createElement('div')
    warning.setAttribute('class','alert alert-danger');
    warning.setAttribute('role','alert');
    warning.innerText = "You missed some fields - fill em' in";

    warningDiv.appendChild(warning);
    resetPar.appendChild(warningDiv);
    console.log(warningDiv);
};

// remove styling
function removeIncompleteFieldStyle(){
    for (var i=0; i<inputFields.length; i++){
        var parent = inputFields[i].parentElement;
        parent.removeAttribute('style');
    };

    try {
        var warningDiv = document.getElementById('warningDiv')
        warningDiv.outerHTML = "";
    } catch (e) {
        console.log(e);
    };
};

function submitButtonClick(){

    var baseURL = 'http://127.0.0.1:5000/';
    var apiBase = 'api/v1.0';

    var nullFields = fieldsNull();

    // if there are null fields
    if (nullFields.length>0) {
        console.log(nullFields.length);
        removeIncompleteFieldStyle();
        addIncompleteFieldStyle(nullFields);

        var container = document.getElementById('inputForm');
    } else { // otherwise find results
        removeIncompleteFieldStyle()

        // create query string by appending all field values to url
        var query = '';
        for (var i=0; i<inputFields.length; i++) {
            var value = inputFields[i].value;
            query = query + '/' + value;

        };
        var fullQuery = baseURL + apiBase + query

        // Create a request variable and assign a new XMLHttpRequest object to it.
        var request = new XMLHttpRequest();
        // Open a new connection, using the GET request on the URL endpoint
        request.open('GET', fullQuery, true);

        request.onload = function () {
            var data = JSON.parse(this.response);
            console.log(data.probability);
            showResult(data.probability);
            }
        // Send request
        request.send();
    };
};

function showResult(proba) {

    var $resultBox = document.getElementById('resultsInner');
    $resultBox.innerHTML = '';

    var resultHeader = document.createElement('div');
    resultHeader.innerText = 'Results'
    resultHeader.setAttribute('id','resultHeader')

    var resultProb = document.createElement('div');
    resultProb.setAttribute('id','resultProb');
    resultProb.innerText = (proba*100).toFixed(2) + '%';

    var resultImg = document.createElement('img');
    resultImg.setAttribute('id','resultImg')
    var resultDesc = document.createElement('p');

    if (proba>=.75) {
        resultProb.setAttribute('class','alert alert-success');
        resultProb.setAttribute('role','alert');
        resultImg.setAttribute('src','../static/images/cheque-yo-self.png')
        resultDesc.innerText = 'Your stats look good. You should be good to go if you apply for a loan!';
    } else if (proba>=.6) {
        resultProb.setAttribute('class','alert alert-warning');
        resultProb.setAttribute('role','alert');
        resultImg.setAttribute('src','../static/images/recheck-yo-self.png')
        resultDesc.innerText = 'We see that your financial future is a tad bit troubling...decreasing your existing debt and increasing your overall credit limit should help.';
    } else {
        resultProb.setAttribute('class','alert alert-danger');
        resultProb.setAttribute('role','alert');
        resultImg.setAttribute('src','../static/images/wreck-yo-self.png')
        resultDesc.innerText = 'Bro...what you been up to? You gotta hide yo kids, hide yo wife. Rebuild that credit son.';
    };

    $resultBox.appendChild(resultHeader);
    $resultBox.appendChild(resultProb);
    $resultBox.appendChild(resultImg);
    $resultBox.appendChild(resultDesc);
};

// <div class="alert alert-success" role="alert">
// <div class="alert alert-danger" role="alert">
// <div class="alert alert-warning" role="alert">

function showRecommendations(){
    var container = document.getElementById('formResults').parentElement;
    var recos = document.createElement('div')
    recos.setAttribute('id','recommendations');

    container.appendChild(recos);
};

// AGE DROPDOWN
var ageArr = createConsecutiveArr(21,95);
renderDropdown(ageArr,$ageInput,'choose');

// INCOME DROPDOWN
var salaryInstuct = [[150000,10000],[300000,25000]];
goMoney(salaryInstuct,$salaryInput);

// DEPENDENTS DROPDOWN
var dependentsArr = createConsecutiveArr(0,20);
renderDropdown(dependentsArr,$dependentsInput);

// CREDIT LINE DROPDOWN
var creditLineInstruct = [[20,1],[61,5]];
var creditLineArr = createSteppedArr(creditLineInstruct,0);
renderDropdown(creditLineArr,$creditLineInput);

// CREDIT LIMIT DROPDOWN
var creditLimitInstruct = [[30000,1000],[200000,10000],[1000000,100000]]
goMoney(creditLimitInstruct,$creditLimitInput);

// REAL ESTATE DROPDOWN
var realEstateInstruct = [[20,1],[61,5]];
var realEstateArr = createSteppedArr(realEstateInstruct,0);
renderDropdown(realEstateArr,$realEstateInput);

// MONTHLY SPEND DROPDOWN
var monthlySpendInstruct = [[5000,250],[10000,500],[20000,1000]];
goMoney(monthlySpendInstruct,$monthlySpendInput);

// TOTAL DEBT DROPDOWN
var totalDebtInstruct = [[5000,500],[10000,1000],[100000,10000],[1000000,150000]];
goMoney(totalDebtInstruct,$totalDebtInput);

// OVERDUE DROPDOWN
var overdueArr = createConsecutiveArr(0,25);
renderDropdown(overdueArr,$overdueInput);

// $submitBtn.addEventListener('click', submitButtonClick);
