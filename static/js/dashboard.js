$('#filter-form').submit(function(e) {
    e.preventDefault();
    let formData = $(this).serialize();

    $.post('/generate_chart', formData, function(data) {
        $('#chart-image').attr('src', data.chart_url + '?t=' + new Date().getTime());
        toastr.success('Chart generated successfully.');
    }).fail(function(err) {
        toastr.error('Failed to generate chart.');
    });
});

$('#download-chart').click(function() {
    const imgsrc = $('#chart-image').attr('src');

    if (imgsrc && !imgsrc.includes('placeholder.png')) {
        const link = document.createElement('a');
        link.href = imgsrc;
        link.download = 'chart.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        toastr.success('Chart downloaded successfully.');
    } else {
        toastr.error('Please generate chart first.');
    }
});
