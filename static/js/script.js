// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('formFile');
    const dropZone = document.getElementById('drop-zone');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultSection = document.getElementById('result-section');
    const noResultsPlaceholder = document.getElementById('no-results-placeholder');
    const uploadedImage = document.getElementById('uploaded-image');
    const resultHeading = document.getElementById('result-heading');
    const diagnosisIcon = document.getElementById('diagnosis-icon');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceLevel = document.getElementById('confidence-level');
    const probabilitiesContainer = document.getElementById('probabilities-container');
    const keyFeatures = document.getElementById('key-features');
    const standardExplanationImg = document.getElementById('standard-explanation-image');
    const deepExplanationImg = document.getElementById('deep-explanation-image');
    const generateExplanation = document.getElementById('generateExplanation');
    const downloadReportBtn = document.getElementById('download-report-btn');
    
    // Loading overlay with countdown
    let countdownInterval;
    let remainingTime = 27; // 27 seconds countdown
    
    function showLoadingOverlay() {
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Add countdown elements if they don't exist
        if (!document.getElementById('countdown-container')) {
            const countdownContainer = document.createElement('div');
            countdownContainer.id = 'countdown-container';
            countdownContainer.style.width = '300px';
            countdownContainer.style.marginTop = '20px';
            
            const countdownText = document.createElement('div');
            countdownText.id = 'countdown-text';
            countdownText.className = 'text-center mb-2';
            countdownText.textContent = `Processing will take approximately ${remainingTime} seconds`;
            
            const progressContainer = document.createElement('div');
            progressContainer.className = 'progress';
            progressContainer.style.height = '10px';
            
            const progressBar = document.createElement('div');
            progressBar.id = 'countdown-progress';
            progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
            progressBar.style.width = '100%';
            
            progressContainer.appendChild(progressBar);
            countdownContainer.appendChild(countdownText);
            countdownContainer.appendChild(progressContainer);
            
            document.querySelector('.loading-spinner').appendChild(countdownContainer);
        }
        
        // Reset countdown
        remainingTime = 26;
        updateCountdown();
        
        // Start countdown
        countdownInterval = setInterval(() => {
            remainingTime--;
            updateCountdown();
            
            if (remainingTime <= 0) {
                clearInterval(countdownInterval);
            }
        }, 1000);
    }
    
    function updateCountdown() {
        const countdownText = document.getElementById('countdown-text');
        const countdownProgress = document.getElementById('countdown-progress');
        
        if (countdownText && countdownProgress) {
            countdownText.textContent = `Processing will complete in approximately ${remainingTime} seconds`;
            countdownProgress.style.width = `${(remainingTime / 26) * 100}%`;
        }
    }
    
    function hideLoadingOverlay() {
        loadingOverlay.style.display = 'none';
        clearInterval(countdownInterval);
    }
    
    // Prevent default behavior for drag events
    ['dragover', 'dragleave', 'drop'].forEach(eventName => {
        if (dropZone) {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        }
    });
    
    // Add active class on dragover
    if (dropZone) {
        dropZone.addEventListener('dragover', () => {
            dropZone.classList.add('active');
        });
        
        // Remove active class on dragleave
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('active');
        });
        
        // Handle file drop
        dropZone.addEventListener('drop', (e) => {
            dropZone.classList.remove('active');
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length) {
                fileInput.files = files;
                updateDropzoneText(files[0].name);
            }
        });
        
        // Handle click on dropzone
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
    }
    
    // Handle file input change
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                updateDropzoneText(fileInput.files[0].name);
            }
        });
    }
    
    // Update dropzone text when file is selected
    function updateDropzoneText(filename) {
        const promptElement = dropZone.querySelector('.drop-zone-prompt');
        if (promptElement) {
            promptElement.textContent = `Selected: ${filename}`;
        }
    }
    
    // Form submission handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!fileInput.files[0]) {
                alert('Please select a file to upload');
                return;
            }
            
            // Show loading overlay with countdown
            showLoadingOverlay();
            
            // Create form data
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('generate_explanation', generateExplanation.checked);
            
            // Send to Flask backend
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading overlay
                hideLoadingOverlay();
                
                if (data.success) {
                    // Process and display results
                    processResults(data);
                } else {
                    // Display error
                    alert('Error: ' + (data.error || 'Unknown error occurred'));
                }
            })
            .catch(error => {
                hideLoadingOverlay();
                alert('An error occurred during processing: ' + error);
            });
        });
    }
    
    // Process and display results
    function processResults(data) {
        if (!data.success) return;
        
        // Hide placeholder, show results
        noResultsPlaceholder.style.display = 'none';
        resultSection.style.display = 'block';
        
        // Set uploaded image
        uploadedImage.src = `/static/uploads/${data.filename}`;
        
        // Set result text
        const resultClass = data.result.predicted_class;
        const isTumor = resultClass !== 'No Tumor';
        
        resultHeading.textContent = resultClass;
        
        // Set diagnosis icon and color
        if (isTumor) {
            diagnosisIcon.innerHTML = '<i class="fas fa-exclamation-circle tumor-positive"></i>';
            resultHeading.className = 'mb-0 text-danger';
            confidenceLevel.className = 'confidence-level bg-danger';
        } else {
            diagnosisIcon.innerHTML = '<i class="fas fa-check-circle tumor-negative"></i>';
            resultHeading.className = 'mb-0 text-success';
            confidenceLevel.className = 'confidence-level bg-success';
        }
        
        // Set confidence
        const confidencePercent = data.result.confidence.toFixed(2);
        confidenceValue.textContent = `${confidencePercent}%`;
        confidenceLevel.style.width = `${confidencePercent}%`;
        
        // Generate key features
        keyFeatures.innerHTML = '';
        if (resultClass === 'Glioma') {
            addKeyFeature('Irregular borders');
            addKeyFeature('Fluid accumulation');
            addKeyFeature('Mid-brain region');
        } else if (resultClass === 'Meningioma') {
            addKeyFeature('Well-defined margins');
            addKeyFeature('Brain surface location');
            addKeyFeature('Homogeneous structure');
        } else if (resultClass === 'Pituitary') {
            addKeyFeature('Small size');
            addKeyFeature('Base of brain');
            addKeyFeature('Sellar region');
        } else {
            addKeyFeature('Normal tissue patterns');
            addKeyFeature('Regular symmetry');
            addKeyFeature('Clear ventricles');
        }
        
        // Set probabilities
        probabilitiesContainer.innerHTML = '';
        Object.entries(data.result.probabilities).forEach(([className, probability]) => {
            const progressDiv = document.createElement('div');
            progressDiv.className = 'probability-item';
            
            const label = document.createElement('div');
            label.className = 'd-flex justify-content-between';
            
            const nameSpan = document.createElement('span');
            nameSpan.textContent = className;
            
            const valueSpan = document.createElement('span');
            valueSpan.textContent = `${probability.toFixed(2)}%`;
            
            label.appendChild(nameSpan);
            label.appendChild(valueSpan);
            
            const progress = document.createElement('div');
            progress.className = 'progress';
            
            const progressBar = document.createElement('div');
            if (className === 'No Tumor') {
                progressBar.className = 'progress-bar bg-success';
            } else {
                progressBar.className = 'progress-bar bg-danger';
            }
            progressBar.style.width = `${probability}%`;
            
            progress.appendChild(progressBar);
            progressDiv.appendChild(label);
            progressDiv.appendChild(progress);
            probabilitiesContainer.appendChild(progressDiv);
        });

        // Handle recommendations based on classification result
        const recommendationSection = document.getElementById('recommendationSection');
        const gliomaRecs = document.getElementById('glioma-recommendations');
        const meningiomaRecs = document.getElementById('meningioma-recommendations');
        const pituitaryRecs = document.getElementById('pituitary-recommendations');
        const noTumorRecs = document.getElementById('no-tumor-recommendations');

        // Hide all recommendation sections first
        gliomaRecs.style.display = 'none';
        meningiomaRecs.style.display = 'none';
        pituitaryRecs.style.display = 'none';
        noTumorRecs.style.display = 'none';

        // Show relevant recommendation based on classification
        if (resultClass === 'Glioma') {
            gliomaRecs.style.display = 'block';
        } else if (resultClass === 'Meningioma') {
            meningiomaRecs.style.display = 'block';
        } else if (resultClass === 'Pituitary') {
            pituitaryRecs.style.display = 'block';
        } else {
            noTumorRecs.style.display = 'block';
        }

        // Update the recommendation title with confidence
        const recommendationTitle = document.getElementById('recommendation-title');
        recommendationTitle.textContent = `Clinical Recommendations (${confidencePercent}% confidence)`;
        
        // Handle explanations
        const explanationSection = document.querySelector('.explanation-section');
        
        if (data.standard_explanation || data.deep_explanation) {
            explanationSection.style.display = 'block';
            
            if (data.standard_explanation) {
                standardExplanationImg.src = `/explanations/${data.standard_explanation}`;
                document.getElementById('standardExplanation').classList.add('show');
            } else {
                document.getElementById('standardExplanation').classList.remove('show');
            }
            
            if (data.deep_explanation) {
                deepExplanationImg.src = `/explanations/${data.deep_explanation}`;
                document.getElementById('deepExplanation').classList.add('show');
            } else {
                document.getElementById('deepExplanation').classList.remove('show');
            }
        } else {
            explanationSection.style.display = 'none';
        }
    }
    
    function addKeyFeature(text) {
        const span = document.createElement('span');
        span.className = 'key-feature-indicator';
        span.innerHTML = `<i class="fas fa-check-circle me-1"></i> ${text}`;
        keyFeatures.appendChild(span);
    }
    
    // Download report button functionality
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', function() {
            // Show a loading message
            const originalButtonText = downloadReportBtn.innerHTML;
            downloadReportBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Generating PDF...';
            downloadReportBtn.disabled = true;
            
            // Get the current date for the report
            const currentDate = new Date();
            const dateString = currentDate.toLocaleDateString();
            const timeString = currentDate.toLocaleTimeString();
            
            // Create a copy of the result section to modify for PDF
            const resultContent = document.querySelector('.result-content').cloneNode(true);
            
            // Create a container for the PDF content
            const pdfContainer = document.createElement('div');
            pdfContainer.style.width = '650px'; // Reduced width to allow for margins
            pdfContainer.style.padding = '40px'; // Increased padding for better margins
            pdfContainer.style.backgroundColor = 'white';
            pdfContainer.style.position = 'absolute';
            pdfContainer.style.left = '-9999px';
            pdfContainer.style.fontFamily = 'Arial, sans-serif';
            
            // Function to create section breaks that help with pagination
            const addSectionBreak = () => {
                const sectionBreak = document.createElement('div');
                sectionBreak.style.pageBreakAfter = 'always';
                sectionBreak.style.marginBottom = '30px';
                sectionBreak.style.height = '1px';
                return sectionBreak;
            };
            
            // Add header with logo and title
            const header = document.createElement('div');
            header.innerHTML = `
                <div style="display: flex; align-items: center; margin-bottom: 25px; border-bottom: 2px solid #3367d6; padding-bottom: 15px;">
                    <div style="font-size: 28px; color: #3367d6; margin-right: 15px;">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div>
                        <h2 style="margin: 0; color: #3367d6; font-size: 24px;">MediExplain AI - Analysis Report</h2>
                        <p style="margin: 5px 0 0 0; color: #666;">Generated on ${dateString} at ${timeString}</p>
                    </div>
                </div>
            `;
            pdfContainer.appendChild(header);
            
            // Get patient/image information
            const imageInfo = document.createElement('div');
            imageInfo.style.marginBottom = '25px';
            
            // Get prediction result
            const resultClass = resultHeading.textContent;
            const confidence = confidenceValue.textContent;
            
            imageInfo.innerHTML = `
                <h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">Analysis Summary</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; border: 1px solid #ddd;">
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; width: 30%; background-color: #f9f9f9;"><strong>Diagnosis:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd; color: ${resultClass !== 'No Tumor' ? '#db4437' : '#0f9d58'};">
                            <strong>${resultClass}</strong>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9;"><strong>Confidence:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">${confidence}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9;"><strong>Analysis Date:</strong></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">${dateString}</td>
                    </tr>
                </table>
            `;
            pdfContainer.appendChild(imageInfo);
            pdfContainer.appendChild(addSectionBreak());

            // Add MRI image section
            const imageSection = document.createElement('div');
            imageSection.style.marginBottom = '25px';
            imageSection.innerHTML = `
                <h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">MRI Image</h3>
                <div style="text-align: center; margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; border-radius: 4px;">
                    <img src="${uploadedImage.src}" style="max-height: 220px; border: 1px solid #ddd; padding: 5px; background-color: white;">
                </div>
            `;
            pdfContainer.appendChild(imageSection);
            
            // Add probability section
            const probabilitySection = document.createElement('div');
            probabilitySection.style.marginBottom = '25px';
            probabilitySection.innerHTML = '<h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">Probability Distribution</h3>';
            
            // Create a table for probabilities instead of progress bars
            const probabilityTable = document.createElement('table');
            probabilityTable.style.width = '100%';
            probabilityTable.style.borderCollapse = 'collapse';
            probabilityTable.style.marginBottom = '20px';
            probabilityTable.style.border = '1px solid #ddd';
            
            // Add table header
            probabilityTable.innerHTML = `
                <tr style="background-color: #f3f3f3;">
                    <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Classification</th>
                    <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Probability</th>
                </tr>
            `;
            
            // Add each probability as a row
            document.querySelectorAll('.probability-item').forEach(item => {
                const className = item.querySelector('.justify-content-between span:first-child')?.textContent || '';
                const probability = item.querySelector('.justify-content-between span:last-child')?.textContent || '';
                
                const row = document.createElement('tr');
                
                // Highlight the detected class with a light background
                if (className === resultClass) {
                    row.style.backgroundColor = className === 'No Tumor' ? '#e8f5e9' : '#fef2f2';
                    row.style.fontWeight = 'bold';
                }
                
                row.innerHTML = `
                    <td style="padding: 10px; border: 1px solid #ddd;">${className}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">${probability}</td>
                `;
                probabilityTable.appendChild(row);
            });
            
            probabilitySection.appendChild(probabilityTable);
            pdfContainer.appendChild(probabilitySection);
            
            // End first page after probability distribution
            pdfContainer.appendChild(addSectionBreak());
            
            // Add key features section (starts on second page)
            const featuresSection = document.createElement('div');
            featuresSection.style.marginBottom = '30px';
            featuresSection.innerHTML = '<h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">Key Features Detected</h3>';
            
            // Create styled list for features
            const featuresList = document.createElement('ul');
            featuresList.style.listStyleType = 'none';
            featuresList.style.padding = '15px';
            featuresList.style.margin = '0';
            featuresList.style.backgroundColor = '#f9f9f9';
            featuresList.style.border = '1px solid #ddd';
            featuresList.style.borderRadius = '4px';
            
            document.querySelectorAll('.key-feature-indicator').forEach(feature => {
                // Extract just the text content, removing the icon
                const featureText = feature.textContent.replace(/^\s*✓\s*/, '').trim();
                const listItem = document.createElement('li');
                listItem.style.padding = '8px 0';
                listItem.style.borderBottom = '1px solid #eee';
                listItem.innerHTML = `<span style="color: ${resultClass !== 'No Tumor' ? '#db4437' : '#0f9d58'}; margin-right: 10px;">✓</span>${featureText}`;
                featuresList.appendChild(listItem);
            });
            
            featuresSection.appendChild(featuresList);
            pdfContainer.appendChild(featuresSection);
            
            // Add recommendations section (still on second page)
            const recommendationsSection = document.createElement('div');
            recommendationsSection.style.marginBottom = '30px';
            recommendationsSection.innerHTML = '<h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">Clinical Recommendations</h3>';
            
            // Get active recommendation content based on diagnosis
            let activeRecommendation;
            if (resultClass === 'Glioma') {
                activeRecommendation = document.getElementById('glioma-recommendations');
            } else if (resultClass === 'Meningioma') {
                activeRecommendation = document.getElementById('meningioma-recommendations');
            } else if (resultClass === 'Pituitary') {
                activeRecommendation = document.getElementById('pituitary-recommendations');
            } else {
                activeRecommendation = document.getElementById('no-tumor-recommendations');
            }
            
            // Clone the recommendation content for the PDF
            if (activeRecommendation) {
                const recommendationContent = activeRecommendation.cloneNode(true);
                
                // Remove any interactive elements or unnecessary styling
                recommendationContent.querySelectorAll('button, .btn').forEach(btn => {
                    btn.remove();
                });
                
                // Add styling for PDF
                recommendationContent.style.display = 'block';
                recommendationContent.style.border = '1px solid #ddd';
                recommendationContent.style.padding = '15px';
                recommendationContent.style.borderRadius = '4px';
                recommendationContent.style.backgroundColor = '#f9f9f9';
                
                // Style text elements inside recommendations
                recommendationContent.querySelectorAll('p').forEach(p => {
                    p.style.margin = '0 0 10px 0';
                    p.style.lineHeight = '1.5';
                });
                
                recommendationContent.querySelectorAll('h4, h5, h6').forEach(h => {
                    h.style.margin = '15px 0 10px 0';
                    h.style.color = '#333';
                });
                
                recommendationContent.querySelectorAll('ul, ol').forEach(list => {
                    list.style.paddingLeft = '25px';
                    list.style.margin = '10px 0';
                });
                
                recommendationContent.querySelectorAll('li').forEach(item => {
                    item.style.margin = '5px 0';
                    item.style.lineHeight = '1.4';
                });
                
                recommendationsSection.appendChild(recommendationContent);
            }
            
            pdfContainer.appendChild(recommendationsSection);
            
            // Add explanation images section - standard SHAP on second page
            const explanationSection = document.createElement('div');
            explanationSection.style.marginBottom = '25px';
            explanationSection.innerHTML = '<h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">Explainability Visualizations</h3>';
            
            // Add standard explanation image if available
            if (standardExplanationImg.src && !standardExplanationImg.src.includes('missing.png')) {
                const standardExplanationDiv = document.createElement('div');
                standardExplanationDiv.style.marginBottom = '15px';
                standardExplanationDiv.innerHTML = `
                    <h4 style="margin-top: 0; margin-bottom: 10px; color: #444; font-size: 16px;">Standard SHAP Explanation</h4>
                    <div style="text-align: center; margin-bottom: 15px;">
                        <img src="${standardExplanationImg.src}" style="max-width: 100%; max-height: 200px; border: 1px solid #ddd; padding: 5px;">
                        <p style="font-size: 12px; color: #666; margin-top: 5px;">
                            SHAP visualization showing regions that influenced the prediction. 
                            Red areas positively contribute to the detected class, while blue areas negatively contribute.
                        </p>
                    </div>
                `;
                explanationSection.appendChild(standardExplanationDiv);
            }
            
            // Add standard SHAP to the explanationSection
            pdfContainer.appendChild(explanationSection);
            
            // End second page after standard SHAP
            pdfContainer.appendChild(addSectionBreak());
            
            // Create deep SHAP section (starts on third page)
            const deepExplanationSection = document.createElement('div');
            deepExplanationSection.style.marginBottom = '25px';
            
            // Add deep explanation image if available
            if (deepExplanationImg.src && !deepExplanationImg.src.includes('missing.png')) {
                deepExplanationSection.innerHTML = `
                    <h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">Deep SHAP Explanation</h3>
                    <div style="text-align: center; margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; border-radius: 4px;">
                        <img src="${deepExplanationImg.src}" style="max-width: 100%; max-height: 300px; border: 1px solid #ddd; padding: 5px; background-color: white;">
                        <p style="font-size: 12px; color: #666; margin-top: 15px;">
                            Deep SHAP visualization providing a more detailed analysis of neural network activations.
                            This view reveals deeper neural network feature contributions to the final classification,
                            highlighting the specific activation patterns that led to the diagnosis.
                        </p>
                    </div>
                `;
                pdfContainer.appendChild(deepExplanationSection);
            }
            
            // Only add the explanation section if at least one image is available
            if ((standardExplanationImg.src && !standardExplanationImg.src.includes('missing.png')) || 
                (deepExplanationImg.src && !deepExplanationImg.src.includes('missing.png'))) {
                pdfContainer.appendChild(explanationSection);
            }
            
            // Add disclaimer (now only as footer on each page)
            const disclaimer = document.createElement('div');
            disclaimer.style.marginTop = '30px';
            disclaimer.style.padding = '15px';
            disclaimer.style.border = '1px solid #f8d7da';
            disclaimer.style.backgroundColor = '#fff3f3';
            disclaimer.style.borderRadius = '4px';
            disclaimer.style.display = 'none'; // Hide it - it will appear in the footer instead
            
            // Add the container to the document temporarily
            document.body.appendChild(pdfContainer);
            
            // Improved PDF generation with better pagination and page numbers
            setTimeout(() => {
                // Instead of using html2canvas for the entire document, we'll create the PDF page by page
                const { jsPDF } = window.jspdf;
                const pdf = new jsPDF('p', 'pt', 'a4');
                
                // Page dimensions
                const pageWidth = pdf.internal.pageSize.getWidth();
                const pageHeight = pdf.internal.pageSize.getHeight();
                const margin = 40;
                const contentWidth = pageWidth - (margin * 2);
                const contentHeight = pageHeight - (margin * 2) - 50; // Extra 50pt for footer
                
                // Footer text
                const disclaimerText = "DISCLAIMER: This report is generated by an AI tool to assist medical professionals and should not be used as the sole basis for diagnosis.";
                let pageNumber = 1;
                
                // Count the total number of pages
                const totalPages = 3; // Fixed structure: 1st: Summary, 2nd: Features+Recommendations+StandardSHAP, 3rd: DeepSHAP
                
                // Function to add footer to each page
                const addFooter = (pageNum) => {
                    // Add page border
                    pdf.setDrawColor(200, 200, 200);
                    pdf.setLineWidth(1);
                    pdf.rect(margin / 2, margin / 2, pageWidth - margin, pageHeight - margin, 'S');
                    
                    // Add disclaimer
                    pdf.setFontSize(8);
                    pdf.setTextColor(128, 28, 36); // Dark red color
                    pdf.text(disclaimerText, pageWidth / 2, pageHeight - (margin / 2) - 15, { align: 'center' }); 
                    
                    // Add page number
                    pdf.setTextColor(100, 100, 100); // Gray color for page number
                    pdf.text(`Page ${pageNum} of ${totalPages}`, pageWidth - margin - 10, pageHeight - (margin / 2) - 10);
                };
                
                // ****** Generate PAGE 1: Analysis summary, MRI image, Probability distribution ******
                const firstPageElements = document.createElement('div');
                firstPageElements.append(
                    header.cloneNode(true),
                    imageInfo.cloneNode(true),
                    imageSection.cloneNode(true),
                    probabilitySection.cloneNode(true)
                );
                firstPageElements.style.width = contentWidth + 'px';
                firstPageElements.style.padding = '0';
                document.body.appendChild(firstPageElements);
                
                // Convert page 1 elements to image and add to PDF
                html2canvas(firstPageElements, {
                    scale: 1.5,
                    useCORS: true,
                    logging: false,
                    width: contentWidth
                }).then(canvas1 => {
                    // Remove temporary elements from body
                    document.body.removeChild(firstPageElements);
                    
                    const imgData1 = canvas1.toDataURL('image/png');
                    
                    // Add the image to the PDF
                    if (canvas1.height * (contentWidth / canvas1.width) <= contentHeight) {
                        // If it fits on one page
                        pdf.addImage(imgData1, 'PNG', margin, margin, contentWidth, canvas1.height * (contentWidth / canvas1.width), null, 'FAST', 0);
                    } else {
                        // If it's taller than one page, scale it to fit
                        const scaleFactor = contentHeight / (canvas1.height * (contentWidth / canvas1.width));
                        pdf.addImage(imgData1, 'PNG', margin, margin, contentWidth * scaleFactor, contentHeight, null, 'FAST', 0);
                    }
                    
                    // Add footer to page 1
                    addFooter(1);
                    
                    // ****** Generate PAGE 2: Key features, Recommendations, Standard SHAP ******
                    pdf.addPage();
                    const secondPageElements = document.createElement('div');
                    
                    // Create a container for page 2 elements
                    secondPageElements.append(
                        featuresSection.cloneNode(true),
                        recommendationsSection.cloneNode(true)
                    );
                    
                    // Only add standard SHAP if it exists
                    if (standardExplanationImg.src && !standardExplanationImg.src.includes('missing.png')) {
                        const standardShapSection = document.createElement('div');
                        standardShapSection.style.marginBottom = '25px';
                        standardShapSection.innerHTML = `
                            <h3 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 18px;">Standard SHAP Explanation</h3>
                            <div style="text-align: center; margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; border-radius: 4px;">
                                <img src="${standardExplanationImg.src}" style="max-width: 100%; max-height: 200px; border: 1px solid #ddd; padding: 5px; background-color: white;">
                                <p style="font-size: 12px; color: #666; margin-top: 5px;">
                                    SHAP visualization showing regions that influenced the prediction. 
                                    Red areas positively contribute to the detected class, while blue areas negatively contribute.
                                </p>
                            </div>
                        `;
                        secondPageElements.appendChild(standardShapSection);
                    }
                    
                    secondPageElements.style.width = contentWidth + 'px';
                    secondPageElements.style.padding = '0';
                    document.body.appendChild(secondPageElements);
                    
                    // Convert page 2 elements to image and add to PDF
                    html2canvas(secondPageElements, {
                        scale: 1.5,
                        useCORS: true,
                        logging: false,
                        width: contentWidth
                    }).then(canvas2 => {
                        // Remove temporary elements
                        document.body.removeChild(secondPageElements);
                        
                        const imgData2 = canvas2.toDataURL('image/png');
                        
                        // Add the image to the PDF (page 2)
                        if (canvas2.height * (contentWidth / canvas2.width) <= contentHeight) {
                            pdf.addImage(imgData2, 'PNG', margin, margin, contentWidth, canvas2.height * (contentWidth / canvas2.width), null, 'FAST', 0);
                        } else {
                            // If it's taller than one page, scale it to fit
                            const scaleFactor = contentHeight / (canvas2.height * (contentWidth / canvas2.width));
                            pdf.addImage(imgData2, 'PNG', margin, margin, contentWidth * scaleFactor, contentHeight, null, 'FAST', 0);
                        }
                        
                        // Add footer to page 2
                        addFooter(2);
                        
                        // ****** Generate PAGE 3: Deep SHAP explanation ******
                        // Only add page 3 if deep SHAP exists
                        if (deepExplanationImg.src && !deepExplanationImg.src.includes('missing.png')) {
                            pdf.addPage();
                            
                            const thirdPageElements = document.createElement('div');
                            const deepShapContent = document.createElement('div');
                            deepShapContent.innerHTML = `
                                <h3 style="margin-top: 0; margin-bottom: 20px; color: #333; font-size: 20px;">Deep SHAP Explanation</h3>
                                <div style="text-align: center; margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; background-color: #f9f9f9; border-radius: 4px;">
                                    <img src="${deepExplanationImg.src}" style="max-width: 100%; max-height: 350px; border: 1px solid #ddd; padding: 5px; background-color: white;">
                                </div>
                                <div style="margin-top: 20px; padding: 15px; border: 1px solid #ddd; background-color: #f5f5f5; border-radius: 4px;">
                                    <p style="font-size: 14px; color: #555; margin: 0; line-height: 1.5;">
                                        Deep SHAP visualization provides a more detailed analysis of neural network activations. 
                                        This advanced view reveals deeper neural network feature contributions to the final classification, 
                                        highlighting the specific activation patterns that led to the diagnosis.
                                    </p>
                                </div>
                            `;
                            
                            thirdPageElements.appendChild(deepShapContent);
                            thirdPageElements.style.width = contentWidth + 'px';
                            thirdPageElements.style.padding = '0';
                            document.body.appendChild(thirdPageElements);
                            
                            // Convert page 3 elements to image and add to PDF
                            html2canvas(thirdPageElements, {
                                scale: 1.5,
                                useCORS: true,
                                logging: false,
                                width: contentWidth
                            }).then(canvas3 => {
                                // Remove temporary elements
                                document.body.removeChild(thirdPageElements);
                                
                                const imgData3 = canvas3.toDataURL('image/png');
                                
                                // Add the image to the PDF (page 3)
                                if (canvas3.height * (contentWidth / canvas3.width) <= contentHeight) {
                                    pdf.addImage(imgData3, 'PNG', margin, margin, contentWidth, canvas3.height * (contentWidth / canvas3.width), null, 'FAST', 0);
                                } else {
                                    // If it's taller than one page, scale it to fit
                                    const scaleFactor = contentHeight / (canvas3.height * (contentWidth / canvas3.width));
                                    pdf.addImage(imgData3, 'PNG', margin, margin, contentWidth * scaleFactor, contentHeight, null, 'FAST', 0);
                                }
                                
                                // Add footer to page 3
                                addFooter(3);
                                
                                // Save PDF
                                pdf.save(`brain-tumor-analysis-${dateString.replace(/\//g, '-')}.pdf`);
                                
                                // Clean up
                                document.body.removeChild(pdfContainer);
                                downloadReportBtn.innerHTML = originalButtonText;
                                downloadReportBtn.disabled = false;
                            });
                        } else {
                            // If no deep SHAP, just save with 2 pages
                            pdf.save(`brain-tumor-analysis-${dateString.replace(/\//g, '-')}.pdf`);
                            
                            // Clean up
                            document.body.removeChild(pdfContainer);
                            downloadReportBtn.innerHTML = originalButtonText;
                            downloadReportBtn.disabled = false;
                        }
                    });
                });
            }, 500);
        });
    }
});