let size = 28;
let grid = document.querySelector("#grid");
let requested = false;
let ongoing = false;
let text_W1;
let text_b1;
let text_W2;
let text_b2;
let W1;
let b1;
let W2;
let b2;
async function init(){
    text_W1 = await getTextFromPath("weight_1");
    text_b1 = await getTextFromPath("bias_1");
    text_W2 = await getTextFromPath("weight_2");
    text_b2 = await getTextFromPath("bias_2");
    W1 = await textToArray(text_W1);
    b1 = await textToArray(text_b1);
    W2 = await textToArray(text_W2);
    b2 = await textToArray(text_b2);
    b1 = b1.flat()
    b2 = b2.flat()
}
init();

function createGrid(){
    let paintWhite = true;
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            let cell = document.createElement("div");
            cell.classList.add("cell");
            cell.style.backgroundColor = paintWhite ? "white" : "blue";
            cell.setAttribute("row", i);
            cell.setAttribute("col", j);
            cell.id = i*28 + j;
            cell.setAttribute("colored", false)
            cell.addEventListener("mouseover", cellHandler);
            paintWhite = !paintWhite;
            grid.appendChild(cell);
        }
        paintWhite = !paintWhite;
    }
}

function cellHandler(e){
    if(e.buttons == 1){
        e.target.style.backgroundColor = "red";
        e.target.setAttribute("colored", true);
        let siblings = e.target.parentNode.childNodes
        let idx = parseInt(e.target.id);
        if(idx%(size) < (size-1)){
            siblings[idx+1].style.backgroundColor = "red"
            siblings[idx+1].setAttribute("colored", true);
        }
        if(idx%size > 0){
            siblings[idx-1].style.backgroundColor = "red"
            siblings[idx-1].setAttribute("colored", true);
        }
        if(idx >= size){
            siblings[idx-size].style.backgroundColor = "red"
            siblings[idx-size].setAttribute("colored", true);
        }
        if(idx <= (size*size-size)){
            siblings[idx+size].style.backgroundColor = "red"
            siblings[idx+size].setAttribute("colored", true);
        }
        requested = true;
        if(!ongoing){
            getData();
        }
    }
}

function ReLU(arr){
    return arr.map((x)=>x>0 ? x : 0);
}


function softmax(arr){
    let a = arr.map((x)=>math.exp(x));
    let s = math.sum(a);
    return a.map((x)=>x/s)
}

async function textToArray(text){
    let arr = [];
    let text_end = text.slice(-2, text.length);
    if(text_end == "\r\n"){
        text = text.slice(0,-2);
    }
    arr = text.split("\n");
    arr = arr.map((s)=>s.split(" ").map((e)=>parseFloat(e)));
    console.log(arr);
    return arr
}

async function getData(){
    let cells = grid.childNodes;
    let data = [];
    cells.forEach((cell)=>data.push(cell.getAttribute("colored")=="true" ? 1 : 0))
    let Z1 = [];
    for (let i = 0; i < W1.length; i++) {
        Z1.push(math.dot(W1[i], data) + b1[i]);    
    }
    let A1 = ReLU(Z1);
    let Z2 = [];
    for (let i = 0; i < W2.length; i++) {
        Z2.push(math.dot(W2[i], A1) + b2[i]);    
    }
    let A2 = softmax(Z2);
    let guess = maxIndex(A2);
    console.log(A2);
    document.querySelector("#guess").textContent = guess;
    ongoing = false;
} 

function logData(){
    let cells = grid.childNodes;
    let data = [];
    cells.forEach((cell)=>data.push(cell.getAttribute("colored")=="true" ? 1 : 0))
    console.log(data.join(" "));
}

function maxIndex(arr){
    let max = arr[0];
    let maxIdx = 0;
    for (let i = 1; i < arr.length; i++) {
        if(arr[i]>max){
            max = arr[i]
            maxIdx = i;
        }
    }
    return maxIdx;
}

function sumArrays(a,b){
    return a.map((num,idx)=>num+b[idx]);
}

async function getTextFromPath(path){  
    let text;
    await fetch(path)
    .then(res => res.text())
    .then(fileText => {
        text = fileText;        
    });
    
    return text;
}

function keyHandler(e){
    if(e.key == "d"){
        getData();
    } else if (e.key == "f"){
        logData();
    }
}
document.addEventListener("keypress", keyHandler)
createGrid()
console.log("done");