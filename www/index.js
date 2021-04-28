$(document).ready(function () {
      
  $.getJSON("/api/profiling", function(table_data) {

    $('#executed-batches').bootstrapTable({
      data: table_data
    });

    $("#executed-batches").delegate('tr', 'click', function() {
      var id = $(this).find("td:first").text();
      window.location.assign("/timeline.html?id=" + id);
    });
  });
});


function dateFormat(value, row, index) {
    return new Date(Math.floor(value / 1000000)).toUTCString()
}