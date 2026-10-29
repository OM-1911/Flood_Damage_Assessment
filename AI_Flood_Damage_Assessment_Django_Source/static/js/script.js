const input = document.getElementById('imageInput');
const nameBox = document.getElementById('fileName');
if (input && nameBox) {
  input.addEventListener('change', () => {
    nameBox.textContent = input.files.length ? input.files[0].name : 'No file selected';
  });
}
