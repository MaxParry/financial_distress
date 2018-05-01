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


function goMoney(instruct,$div) {
    var arr = createSteppedArr(instruct,1);
    var styledArr = showMeTheMoney(arr);
    var zippedArr = zipIt(arr,styledArr);
    renderDropdown(zippedArr,$div,'choose');
};

function submitButtonClick(){

    var baseURL = 'http://127.0.0.1:5000/';
    var apiBase = 'api/v1.0';
    var query = '';
    for (var i=0; i<inputFields.length; i++) {
        var value = inputFields[i].value;
        query = query + '/' + value;
    };
    console.log(query);
    var fullQuery = baseURL + apiBase + query

    console.log(fullQuery);

    // Create a request variable and assign a new XMLHttpRequest object to it.
    var request = new XMLHttpRequest();
    // Open a new connection, using the GET request on the URL endpoint
    request.open('GET', fullQuery, true);

    request.onload = function () {
        var data = JSON.parse(this.response);
        console.log(data);
        }
    // Send request
    request.send();

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
