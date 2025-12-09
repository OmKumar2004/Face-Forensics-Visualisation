async function loadData() {
  try {
    const res = await fetch('../descriptions.json');
    if (!res.ok) throw new Error('Failed to load descriptions.json');
    const data = await res.json();
    initUI(data);
  } catch (err) {
    console.error(err);
    document.getElementById('description').textContent =
      'Error loading data: ' + err.message;
  }
}

function initUI(items) {
  const select = document.getElementById('imageSelect');
  const descEl = document.getElementById('description');
  const imgEl = document.getElementById('mainImage');

  // Populate dropdown with image names
  items.forEach((item, idx) => {
    const opt = document.createElement('option');
    opt.value = item.image_name;
    opt.textContent = `${idx + 1}. ${item.image_name}`;
    select.appendChild(opt);
  });

  function showItem(imageName) {
    const item = items.find(i => i.image_name === imageName);
    if (!item) return;

    // Image path relative to /docs/index.html
    imgEl.src = `../final_dataset_images/${item.image_name}`;
    imgEl.alt = item.image_name;

    // combined_facial may be a JSON string or plain text; normalise
    let txt = item.combined_facial;
    try {
      const parsed = JSON.parse(item.combined_facial);
      txt = JSON.stringify(parsed, null, 2);
    } catch (e) {
      // leave as-is if not valid JSON
    }
    descEl.textContent = txt;
  }

  select.addEventListener('change', e => {
    showItem(e.target.value);
  });

  if (items.length > 0) {
    select.value = items[0].image_name;
    showItem(items[0].image_name);
  }
}

loadData();
